import dataclasses
import weakref
import math
from pathlib import Path

import imgui
import lietorch
import torch
import moderngl
import moderngl_window as mglw
import numpy as np
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.pose_utils import translation_matrix
from in3d.color import hex2rgba
from in3d.geometry import Axis
from in3d.viewport_window import ViewportWindow
from in3d.window import WindowEvents
from in3d.image import Image
from moderngl_window import resources
from moderngl_window.timers.clock import Timer

from splatt3r_slam.frame import Mode
from splatt3r_slam.geometry import get_pixel_coords
from splatt3r_slam.lietorch_utils import as_SE3
from splatt3r_slam.visualization_utils import (
    Frustums,
    Lines,
    depth2rgb,
    image_with_text,
)
from splatt3r_slam.config import load_config, config, set_global_config

# Gaussian rasterization (same CUDA rasterizer used by DecoderSplattingCUDA)
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    _HAS_DIFF_GS = True
except ImportError:
    _HAS_DIFF_GS = False
    print(
        "[viz] diff_gaussian_rasterization not found – GS interactive rendering disabled"
    )


@dataclasses.dataclass
class WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5
    spatial_stride: int = 4
    max_gaussians: int = 4 * 1024 * 1024
    # Splash-artifact filter params (synced via GUI sliders)
    depth_max_percentile: float = 0.98
    max_scale: float = 1.0
    min_confidence: float = 1.5
    keep_ratio: float = 0.6


class Window(WindowEvents):
    title = "Splatt3R-SLAM"
    window_size = (1960, 1080)

    def __init__(
        self,
        states,
        keyframes,
        shared_gaussians,
        main2viz,
        viz2main,
        spatial_stride=4,
        max_gaussians=4 * 1024 * 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"
        # bit hacky, but detect whether user is using 4k monitor
        self.scale = 1.0
        if self.wnd.buffer_size[0] > 2560:
            self.set_font_scale(2.0)
            self.scale = 2
        self.clear = hex2rgba("#1E2326", alpha=1)
        resources.register_dir((Path(__file__).parent.parent / "resources").resolve())

        self.line_prog = self.load_program("programs/lines.glsl")
        self.surfelmap_prog = self.load_program("programs/surfelmap.glsl")
        self.trianglemap_prog = self.load_program("programs/trianglemap.glsl")
        self.pointmap_prog = self.surfelmap_prog

        width, height = self.wnd.size
        self.camera = Camera(
            ProjectionMatrix(width, height, 60, width // 2, height // 2, 0.05, 100),
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        )
        self.axis = Axis(self.line_prog, 0.1, 3 * self.scale)
        self.frustums = Frustums(self.line_prog)
        self.lines = Lines(self.line_prog)

        self.viewport = ViewportWindow("Scene", self.camera)
        self.state = WindowMsg(
            spatial_stride=spatial_stride,
            max_gaussians=max_gaussians,
            depth_max_percentile=kwargs.get("depth_max_percentile", 0.98),
            max_scale=kwargs.get("max_scale", 1.0),
            min_confidence=kwargs.get("min_confidence", 1.5),
            keep_ratio=kwargs.get("keep_ratio", 0.6),
        )
        self.keyframes = keyframes
        self.states = states
        self.shared_gaussians = shared_gaussians

        self.show_all = True
        self.show_keyframe_edges = True
        self.culling = True
        self.follow_cam = True

        self.depth_bias = 0.001
        self.frustum_scale = 0.05

        self.dP_dz = None

        self.line_thickness = 3
        self.show_keyframe = True
        self.show_curr_pointmap = True
        self.show_axis = True

        self.textures = dict()
        self.mtime = self.pointmap_prog.extra["meta"].resolved_path.stat().st_mtime
        self.curr_img, self.kf_img = Image(), Image()
        self.curr_img_np, self.kf_img_np = None, None

        self.main2viz = main2viz
        self.viz2main = viz2main

        # CLI defaults for GUI sliders
        self.spatial_stride = spatial_stride
        self.max_gaussians_limit = max_gaussians  # buffer allocation cap
        self.max_gaussians = max_gaussians  # current effective value
        # Splash-filter defaults (may be overridden by kwargs from CLI)
        self.depth_max_percentile = kwargs.get("depth_max_percentile", 0.98)
        self.max_scale = kwargs.get("max_scale", 1.0)
        self.min_confidence_gs = kwargs.get("min_confidence", 1.5)

        # --- Gaussian Splatting interactive rendering ---
        self.use_gs_rendering = _HAS_DIFF_GS  # ON by default when available
        self.gs_render_img = Image()  # preview image for GUI panel
        self.gs_tex = None  # moderngl texture for fullscreen quad
        self.gs_resolution_scale = 0.5  # render at fraction of viewport size
        # Fullscreen quad shader for displaying GS-rendered images
        self.gs_quad_prog = self.ctx.program(
            vertex_shader="""
            #version 330 core
            out vec2 uv;
            void main() {
                float x = float(gl_VertexID % 2) * 2.0 - 1.0;
                float y = float(gl_VertexID / 2) * 2.0 - 1.0;
                gl_Position = vec4(x, y, 0.0, 1.0);
                uv = vec2((x + 1.0) * 0.5, (-y + 1.0) * 0.5);
            }
            """,
            fragment_shader="""
            #version 330 core
            uniform sampler2D gs_texture;
            in vec2 uv;
            out vec4 fragColor;
            void main() {
                fragColor = vec4(texture(gs_texture, uv).rgb, 1.0);
            }
            """,
        )
        self.gs_quad_vao = self.ctx.vertex_array(self.gs_quad_prog, [])

    def render(self, t: float, frametime: float):
        self.viewport.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        if self.culling:
            self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.clear(*self.clear)

        # --- Interactive Gaussian Splatting rendering from shared buffer ---
        gs_active = self.use_gs_rendering and _HAS_DIFF_GS
        if gs_active:
            gs_img_np = self._render_gs_interactive()
            if gs_img_np is not None:
                gs_active = True
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ctx.disable(moderngl.CULL_FACE)
                self._render_gs_fullscreen(gs_img_np)
                self.gs_render_img.write(gs_img_np)
                self.ctx.enable(moderngl.DEPTH_TEST)
                if self.culling:
                    self.ctx.enable(moderngl.CULL_FACE)
            else:
                gs_active = False

        self.ctx.point_size = 2
        if self.show_axis:
            self.axis.render(self.camera)

        curr_frame = self.states.get_frame()
        h, w = curr_frame.img_shape.flatten()
        self.frustums.make_frustum(h, w)

        self.curr_img_np = curr_frame.uimg.numpy()
        self.curr_img.write(self.curr_img_np)

        cam_T_WC = as_SE3(curr_frame.T_WC).cpu()
        if self.follow_cam:
            T_WC = cam_T_WC.matrix().numpy().astype(
                dtype=np.float32
            ) @ translation_matrix(np.array([0, 0, -2], dtype=np.float32))
            self.camera.follow_cam(np.linalg.inv(T_WC))
        else:
            self.camera.unfollow_cam()
        self.frustums.add(
            cam_T_WC,
            scale=self.frustum_scale,
            color=[0, 1, 0, 1],
            thickness=self.line_thickness * self.scale,
        )

        with self.keyframes.lock:
            N_keyframes = len(self.keyframes)
            dirty_idx = self.keyframes.get_dirty_idx()

        for kf_idx in dirty_idx:
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            X = self.frame_X(keyframe)
            C = keyframe.get_average_conf().cpu().numpy().astype(np.float32)

            if keyframe.frame_id not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures[keyframe.frame_id] = ptex, ctex, itex
                ptex, ctex, itex = self.textures[keyframe.frame_id]
                itex.write(keyframe.uimg.numpy().astype(np.float32).tobytes())

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())

        for kf_idx in range(N_keyframes):
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            if kf_idx == N_keyframes - 1:
                self.kf_img_np = keyframe.uimg.numpy()
                self.kf_img.write(self.kf_img_np)

            color = [1, 0, 0, 1]
            if self.show_keyframe:
                self.frustums.add(
                    as_SE3(keyframe.T_WC.cpu()),
                    scale=self.frustum_scale,
                    color=color,
                    thickness=self.line_thickness * self.scale,
                )

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            if self.show_all and not gs_active:
                self.render_pointmap(keyframe.T_WC.cpu(), w, h, ptex, ctex, itex)

        if self.show_keyframe_edges:
            with self.states.lock:
                ii = torch.tensor(self.states.edges_ii, dtype=torch.long)
                jj = torch.tensor(self.states.edges_jj, dtype=torch.long)
                if ii.numel() > 0 and jj.numel() > 0:
                    T_WCi = lietorch.Sim3(self.keyframes.T_WC[ii, 0])
                    T_WCj = lietorch.Sim3(self.keyframes.T_WC[jj, 0])
            if ii.numel() > 0 and jj.numel() > 0:
                t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
                t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()
                self.lines.add(
                    t_WCi,
                    t_WCj,
                    thickness=self.line_thickness * self.scale,
                    color=[0, 1, 0, 1],
                )
        if (
            self.show_curr_pointmap
            and not gs_active
            and self.states.get_mode() != Mode.INIT
        ):
            if config["use_calib"]:
                curr_frame.K = self.keyframes.get_intrinsics()
            h, w = curr_frame.img_shape.flatten()
            X = self.frame_X(curr_frame)
            C = curr_frame.C.cpu().numpy().astype(np.float32)
            if "curr" not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures["curr"] = ptex, ctex, itex
            ptex, ctex, itex = self.textures["curr"]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())
            itex.write(depth2rgb(X[..., -1], colormap="turbo"))
            self.render_pointmap(
                curr_frame.T_WC.cpu(),
                w,
                h,
                ptex,
                ctex,
                itex,
                use_img=True,
                depth_bias=self.depth_bias,
            )

        self.lines.render(self.camera)
        self.frustums.render(self.camera)
        self.render_ui()

    def render_ui(self):
        self.wnd.use()
        imgui.new_frame()

        io = imgui.get_io()
        # get window size and full screen
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()

        imgui.set_next_window_size(
            window_size[0] / 4, 15 * window_size[1] / 16, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_position(
            32 * self.scale, 32 * self.scale, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_focus()
        imgui.begin("GUI", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        new_state = WindowMsg()
        _, new_state.is_paused = imgui.checkbox("pause", self.state.is_paused)

        imgui.spacing()
        _, new_state.C_conf_threshold = imgui.slider_float(
            "C_conf_threshold", self.state.C_conf_threshold, 0, 10
        )

        imgui.spacing()

        _, self.show_all = imgui.checkbox("show all", self.show_all)
        imgui.same_line()
        _, self.follow_cam = imgui.checkbox("follow cam", self.follow_cam)

        imgui.spacing()
        _, new_state.spatial_stride = imgui.slider_int(
            "spatial stride", self.state.spatial_stride, 1, 16
        )
        # Allow slider to go up to 2x the CLI allocation (buffer will FIFO-evict)
        max_gs_slider_upper = max(self.max_gaussians_limit, 8 * 1024 * 1024) // 1024
        _, new_state.max_gaussians = imgui.slider_int(
            "max gaussians (k)",
            self.state.max_gaussians // 1024,
            64,
            max_gs_slider_upper,
        )
        new_state.max_gaussians *= 1024
        imgui.spacing()

        # --- Splash-artifact filter sliders ---
        imgui.separator()
        imgui.text("Splash Filter")
        _, new_state.depth_max_percentile = imgui.slider_float(
            "depth max pct", self.state.depth_max_percentile, 0.5, 1.0
        )
        _, new_state.max_scale = imgui.slider_float(
            "max scale", self.state.max_scale, 0.01, 3.0
        )
        _, new_state.min_confidence = imgui.slider_float(
            "min confidence", self.state.min_confidence, 0.0, 10.0
        )
        _, new_state.keep_ratio = imgui.slider_float(
            "keep ratio", self.state.keep_ratio, 0.1, 1.0
        )
        imgui.separator()
        imgui.spacing()
        if _HAS_DIFF_GS:
            _, self.use_gs_rendering = imgui.checkbox(
                "GS rendering (Splatt3R)", self.use_gs_rendering
            )
            if self.use_gs_rendering:
                _, self.gs_resolution_scale = imgui.slider_float(
                    "GS resolution", self.gs_resolution_scale, 0.1, 1.0
                )
                gs_data = self.shared_gaussians.get_all()
                n_gs = 0 if gs_data is None else gs_data[0].shape[0]
                imgui.text(f"Gaussians: {n_gs:,}")
        imgui.spacing()

        # Point-cloud shader options (only relevant when GS rendering is off)
        if not self.use_gs_rendering:
            shader_options = [
                "surfelmap.glsl",
                "trianglemap.glsl",
            ]
            current_shader = shader_options.index(
                self.pointmap_prog.extra["meta"].resolved_path.name
            )

            for i, shader in enumerate(shader_options):
                if imgui.radio_button(shader, current_shader == i):
                    current_shader = i

            selected_shader = shader_options[current_shader]
            if selected_shader != self.pointmap_prog.extra["meta"].resolved_path.name:
                self.pointmap_prog = self.load_program(f"programs/{selected_shader}")

            imgui.spacing()

            _, self.show_keyframe_edges = imgui.checkbox(
                "show_keyframe_edges", self.show_keyframe_edges
            )
            imgui.spacing()

            _, self.pointmap_prog["show_normal"].value = imgui.checkbox(
                "show_normal", self.pointmap_prog["show_normal"].value
            )
            imgui.same_line()
            _, self.culling = imgui.checkbox("culling", self.culling)
            if "radius" in self.pointmap_prog:
                _, self.pointmap_prog["radius"].value = imgui.drag_float(
                    "radius",
                    self.pointmap_prog["radius"].value,
                    0.0001,
                    min_value=0.0,
                    max_value=0.1,
                )
            if "slant_threshold" in self.pointmap_prog:
                _, self.pointmap_prog["slant_threshold"].value = imgui.drag_float(
                    "slant_threshold",
                    self.pointmap_prog["slant_threshold"].value,
                    0.1,
                    min_value=0.0,
                    max_value=1.0,
                )
            _, self.show_curr_pointmap = imgui.checkbox(
                "show_curr_pointmap", self.show_curr_pointmap
            )
        else:
            _, self.show_keyframe_edges = imgui.checkbox(
                "show_keyframe_edges", self.show_keyframe_edges
            )
            imgui.spacing()
        _, self.show_keyframe = imgui.checkbox("show_keyframe", self.show_keyframe)
        _, self.show_axis = imgui.checkbox("show_axis", self.show_axis)
        _, self.line_thickness = imgui.drag_float(
            "line_thickness", self.line_thickness, 0.1, 10, 0.5
        )

        _, self.frustum_scale = imgui.drag_float(
            "frustum_scale", self.frustum_scale, 0.001, 0, 0.1
        )

        imgui.spacing()

        gui_size = imgui.get_content_region_available()
        scale = gui_size[0] / self.curr_img.texture.size[0]
        scale = min(self.scale, scale)
        size = (
            self.curr_img.texture.size[0] * scale,
            self.curr_img.texture.size[1] * scale,
        )
        image_with_text(self.gs_render_img, size, "gs_render", same_line=False)
        image_with_text(self.kf_img, size, "kf", same_line=False)
        image_with_text(self.curr_img, size, "curr", same_line=False)

        imgui.end()

        if new_state != self.state:
            self.state = new_state
            self.send_msg()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def send_msg(self):
        self.viz2main.put(self.state)

    def _render_gs_fullscreen(self, gs_img_np):
        """Render gaussian-splatted image as fullscreen background quad.

        Args:
            gs_img_np: (H, W, 3) float32 numpy array in [0, 1].
        """
        h, w = gs_img_np.shape[:2]
        if self.gs_tex is None or self.gs_tex.size != (w, h):
            if self.gs_tex is not None:
                self.gs_tex.release()
            self.gs_tex = self.ctx.texture((w, h), 3, dtype="f4")
            self.gs_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.gs_tex.write(gs_img_np.astype(np.float32).tobytes())
        self.gs_tex.use(0)
        self.gs_quad_prog["gs_texture"].value = 0
        self.gs_quad_vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4)

    @torch.inference_mode()
    def _render_gs_interactive(self):
        """Render Gaussians from SharedGaussians buffer using the interactive camera.

        Uses diff_gaussian_rasterization to rasterize all accumulated Gaussians
        from the current interactive viewport camera.  Returns (H, W, 3) float32
        numpy array in [0, 1] or None if no Gaussians are available.
        """
        gs_data = self.shared_gaussians.get_all()
        if gs_data is None:
            return None

        try:
            means, cov_triu, colors, opacities = gs_data

            # --- Viewport size (scaled for performance) ---
            vp_w, vp_h = self.camera.viewport_size
            scale = max(0.1, min(1.0, self.gs_resolution_scale))
            render_w = max(64, int(vp_w * scale))
            render_h = max(64, int(vp_h * scale))

            # --- Camera matrices ---
            # self.camera.T_CW is world-to-camera (OpenGL convention: y-up, z-back)
            # render_cuda expects camera-to-world (T_WC) extrinsics.
            # self.camera.T_CW is world-to-camera in OpenGL convention (y-up, z-back).
            # Convert to OpenCV convention (y-down, z-forward) then invert to get T_WC.
            cv2gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
            T_CW_gl = self.camera.T_CW.astype(np.float32)  # (4, 4) OpenGL
            T_CW_cv = cv2gl @ T_CW_gl  # (4, 4) OpenCV convention  (world-to-camera)
            T_WC_cv = np.linalg.inv(T_CW_cv)  # (4, 4) camera-to-world

            extrinsics = (
                torch.from_numpy(T_WC_cv).unsqueeze(0).to(means.device)
            )  # (1, 4, 4)

            # --- Intrinsics (normalised by image dims, as expected by get_fov) ---
            # in3d uses glm.perspective(radians(hfov/2), ...) so actual vfov = hfov / 2
            vfov_deg = self.camera.proj_mat.hfov / 2.0  # actual vertical FOV in degrees
            vfov_rad = math.radians(vfov_deg)
            # Compute focal lengths in pixels from vfov and render size
            fy = render_h / (2.0 * math.tan(vfov_rad / 2.0))
            fx = fy  # square pixels assumption
            cx, cy = render_w / 2.0, render_h / 2.0
            # normalised intrinsics (divide by image dims)
            K_norm = torch.tensor(
                [
                    [fx / render_w, 0, cx / render_w],
                    [0, fy / render_h, cy / render_h],
                    [0, 0, 1],
                ],
                dtype=torch.float32,
                device=means.device,
            ).unsqueeze(
                0
            )  # (1, 3, 3)

            near = torch.tensor([0.05], device=means.device, dtype=torch.float32)
            far = torch.tensor([100.0], device=means.device, dtype=torch.float32)
            bg = torch.tensor(
                [[0.118, 0.137, 0.149]], device=means.device
            )  # match clear color

            # --- Build projection matrices (same pipeline as render_cuda) ---
            from splatt3r_core.src.pixelsplat_src.projection import get_fov

            fov_xy = get_fov(K_norm)  # (1, 2)
            fov_x, fov_y = fov_xy[0, 0], fov_xy[0, 1]
            tan_fov_x = (0.5 * fov_x).tan().item()
            tan_fov_y = (0.5 * fov_y).tan().item()

            # Scale-invariant mode: scale = 1/near
            inv_near = 1.0 / near
            ext = extrinsics.clone()
            ext[..., :3, 3] *= inv_near[:, None]
            sc_means = means * inv_near.item()
            sc_cov_factor = (inv_near**2).item()
            sc_near = near * inv_near
            sc_far = far * inv_near

            from splatt3r_core.src.pixelsplat_src.cuda_splatting import (
                get_projection_matrix,
            )

            proj = get_projection_matrix(
                sc_near, sc_far, fov_x.unsqueeze(0), fov_y.unsqueeze(0)
            )  # (1,4,4)
            proj_t = proj[0].T  # column-major
            view_t = ext[0].inverse().T  # column-major
            full_proj = view_t @ proj_t

            # --- Rasterize ---
            n_gauss = means.shape[0]
            mean_grads = torch.zeros(
                n_gauss, 3, device=means.device, requires_grad=False
            )

            settings = GaussianRasterizationSettings(
                image_height=render_h,
                image_width=render_w,
                tanfovx=tan_fov_x,
                tanfovy=tan_fov_y,
                bg=bg[0],
                scale_modifier=1.0,
                viewmatrix=view_t,
                projmatrix=full_proj,
                sh_degree=0,
                campos=ext[0, :3, 3],
                prefiltered=False,
                debug=False,
            )
            rasterizer = GaussianRasterizer(settings)

            # Reconstruct covariance from upper-triangle, then apply scale-invariant factor
            row, col = torch.triu_indices(3, 3)
            cov_3x3 = torch.zeros(n_gauss, 3, 3, device=means.device)
            cov_3x3[:, row, col] = cov_triu
            cov_3x3[:, col, row] = cov_triu  # symmetrise
            cov_3x3 = cov_3x3 * sc_cov_factor
            cov_precomp = cov_3x3[:, row, col]  # (G, 6)

            image, _radii = rasterizer(
                means3D=sc_means,
                means2D=mean_grads,
                shs=None,
                colors_precomp=colors,
                opacities=opacities.unsqueeze(-1),
                cov3D_precomp=cov_precomp,
            )
            # image: (3, H, W)
            result = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
            return np.ascontiguousarray(result)
        except Exception as e:
            print(f"[GS render] Error: {e}")
            return None

    def render_pointmap(self, T_WC, w, h, ptex, ctex, itex, use_img=True, depth_bias=0):
        w, h = int(w), int(h)
        ptex.use(0)
        ctex.use(1)
        itex.use(2)
        model = T_WC.matrix().numpy().astype(np.float32).T

        vao = self.ctx.vertex_array(self.pointmap_prog, [], skip_errors=True)
        vao.program["m_camera"].write(self.camera.gl_matrix())
        vao.program["m_model"].write(model)
        vao.program["m_proj"].write(self.camera.proj_mat.gl_matrix())

        vao.program["pointmap"].value = 0
        vao.program["confs"].value = 1
        vao.program["img"].value = 2
        vao.program["width"].value = w
        vao.program["height"].value = h
        vao.program["conf_threshold"] = self.state.C_conf_threshold
        vao.program["use_img"] = use_img
        if "depth_bias" in self.pointmap_prog:
            vao.program["depth_bias"] = depth_bias
        vao.render(mode=moderngl.POINTS, vertices=w * h)
        vao.release()

    def frame_X(self, frame):
        if config["use_calib"]:
            Xs = frame.X_canon[None]
            if self.dP_dz is None:
                device = Xs.device
                dtype = Xs.dtype
                img_size = frame.img_shape.flatten()[:2]
                K = frame.K
                p = get_pixel_coords(
                    Xs.shape[0], img_size, device=device, dtype=dtype
                ).view(*Xs.shape[:-1], 2)
                tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
                tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
                self.dP_dz = torch.empty(
                    p.shape[:-1] + (3, 1), device=device, dtype=dtype
                )
                self.dP_dz[..., 0, 0] = tmp1
                self.dP_dz[..., 1, 0] = tmp2
                self.dP_dz[..., 2, 0] = 1.0
                self.dP_dz = self.dP_dz[..., 0].cpu().numpy().astype(np.float32)
            return (Xs[..., 2:3].cpu().numpy().astype(np.float32) * self.dP_dz)[0]

        return frame.X_canon.cpu().numpy().astype(np.float32)


def run_visualization(
    cfg,
    states,
    keyframes,
    shared_gaussians,
    main2viz,
    viz2main,
    spatial_stride=4,
    max_gaussians=4 * 1024 * 1024,
    **extra_kwargs,
) -> None:
    set_global_config(cfg)

    config_cls = Window
    backend = "glfw"
    window_cls = mglw.get_local_window_cls(backend)

    window = window_cls(
        title=config_cls.title,
        size=config_cls.window_size,
        fullscreen=False,
        resizable=True,
        visible=True,
        gl_version=(3, 3),
        aspect_ratio=None,
        vsync=True,
        samples=4,
        cursor=True,
        backend=backend,
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    window.ctx.gc_mode = "auto"
    timer = Timer()
    window_config = config_cls(
        states=states,
        keyframes=keyframes,
        shared_gaussians=shared_gaussians,
        main2viz=main2viz,
        viz2main=viz2main,
        spatial_stride=spatial_stride,
        max_gaussians=max_gaussians,
        ctx=window.ctx,
        wnd=window,
        timer=timer,
        **extra_kwargs,
    )
    # Avoid the event assigning in the property setter for now
    # We want the even assigning to happen in WindowConfig.__init__
    # so users are free to assign them in their own __init__.
    window._config = weakref.ref(window_config)

    # Swap buffers once before staring the main loop.
    # This can trigged additional resize events reporting
    # a more accurate buffer size
    window.swap_buffers()
    window.set_default_viewport()

    timer.start()

    while not window.is_closing:
        current_time, delta = timer.next_frame()

        if window_config.clear_color is not None:
            window.clear(*window_config.clear_color)

        # Always bind the window framebuffer before calling render
        window.use()

        window.render(current_time, delta)
        if not window.is_closing:
            window.swap_buffers()

    state = window_config.state
    window.destroy()
    state.is_terminated = True
    viz2main.put(state)
