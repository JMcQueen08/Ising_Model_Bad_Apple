from pathlib import Path
from math import exp

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from numba import njit
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    sobel,
)
from tqdm import tqdm


@njit(cache=True)
def _update_checkerboard(spins, magnetic_field, temperature, random_values, parity, coupling):
    size = spins.shape[0]
    neighbour_sum = np.empty(spins.shape, dtype=np.int8)

    for row in range(size):
        up = row - 1 if row > 0 else size - 1
        down = row + 1 if row + 1 < size else 0
        for col in range(size):
            left = col - 1 if col > 0 else size - 1
            right = col + 1 if col + 1 < size else 0
            neighbour_sum[row, col] = (
                spins[up, col] + spins[down, col] + spins[row, left] + spins[row, right]
            )

    accepted = 0
    for row in range(size):
        for col in range(size):
            if (row + col) % 2 == parity:
                delta_energy = 2.0 * spins[row, col] * (
                    coupling * neighbour_sum[row, col] + magnetic_field[row, col]
                )
                if delta_energy <= 0.0 or random_values[row, col] < exp(
                    -delta_energy / temperature[row, col]
                ):
                    spins[row, col] *= -1
                    accepted += 1

    return accepted


class BadAppleIsing:
    def __init__(self, video_path: str, N: int = 150, J: float = 1.0, H_strength: float = 2.5, distance_scale: float = 5.0,
                 T_bulk: float = 1.35, T_edge: float = 2.55, edge_width: float = 5.0, spatial_blur: float = 1.5,
                 temporal_smoothing: float = 0.18, field_spatial_smoothing: float = 0.75,
                 sweeps_per_frame: int = 8, initial_sweeps: int | None = None,
                 invert_video: bool = False, seed: int | None = None):

        if initial_sweeps is None:
            initial_sweeps = sweeps_per_frame * 10

        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)

        self.N = N
        self.J = J
        self.H_strength = H_strength
        self.distance_scale = distance_scale
        self.T_bulk = T_bulk
        self.T_edge = T_edge
        self.edge_width = edge_width
        self.spatial_blur = spatial_blur
        self.temporal_smoothing = temporal_smoothing
        self.field_spatial_smoothing = field_spatial_smoothing
        self.sweeps_per_frame = sweeps_per_frame
        self.initial_sweeps = initial_sweeps
        self.invert_video = invert_video
        self.has_previous_frame = False

        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.choice([-1, 1], size=(N, N)).astype(np.int8)

        rows, cols = np.indices((N, N))
        self.black = (rows + cols) % 2 == 0
        self.white = ~self.black

        self.H = np.zeros((N, N), dtype=np.float32)
        self.T = np.full((N, N), T_bulk, dtype=np.float32)

        self.cap = cv2.VideoCapture(str(self.video_path))
        self.input_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.surface_step = 3
        coords = np.arange(0, N, self.surface_step)
        self.X_surface, self.Y_surface = np.meshgrid(coords, coords)

        self.animation = None
        self.fig = None
        self.spin_image = None
        self.title = None
        self.temperature_surface = None
        self.field_surface = None


    def read_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        side = min(height, width)
        y0 = (height - side) // 2
        x0 = (width - side) // 2
        gray = gray[y0:y0+side, x0:x0+side]

        gray = cv2.resize(gray, (self.N, self.N), interpolation=cv2.INTER_AREA)
        image = gray.astype(np.float32) / 255.0

        if self.invert_video:
            image = 1.0 - image

        return image


    def construct_fields(self, image):
        smoothed_image = gaussian_filter(image, sigma=self.spatial_blur, mode="wrap")
        binary = smoothed_image >= 0.5

        if binary.all():
            signed_distance = np.full(binary.shape, self.N, dtype=np.float32)
            edge_strength = np.zeros(binary.shape, dtype=np.float32)

        elif not binary.any():
            signed_distance = np.full(binary.shape, -self.N, dtype=np.float32)
            edge_strength = np.zeros(binary.shape, dtype=np.float32)

        else:
            tiled_binary = np.tile(binary, (3, 3))
            tiled_inside = distance_transform_edt(tiled_binary)
            tiled_outside = distance_transform_edt(~tiled_binary)
            center = np.s_[self.N:2 * self.N, self.N:2 * self.N]
            signed_distance = tiled_inside[center] - tiled_outside[center]

        target_H = self.H_strength * np.tanh(signed_distance / self.distance_scale)

        gx = sobel(smoothed_image, axis=1, mode="wrap")
        gy = sobel(smoothed_image, axis=0, mode="wrap")

        gradient = np.hypot(gx, gy)
        gradient_max = gradient.max()
        if gradient_max > 0:
            gradient /= gradient_max

        edge_distance = np.abs(signed_distance)
        distance_edge = np.exp(-((edge_distance / self.edge_width) ** 2))

        if not (binary.all() or not binary.any()):
            edge_strength = np.maximum(distance_edge, gradient)
            edge_strength = gaussian_filter(edge_strength, sigma=1.5, mode="wrap")
            edge_strength = np.clip(edge_strength, 0.0, 1.0)

        target_T = self.T_bulk + (self.T_edge - self.T_bulk) * edge_strength

        if self.field_spatial_smoothing > 0:
            target_H = gaussian_filter(target_H, sigma=self.field_spatial_smoothing, mode="wrap")
            target_T = gaussian_filter(target_T, sigma=self.field_spatial_smoothing, mode="wrap")

        if self.has_previous_frame:
            alpha = self.temporal_smoothing
            self.H = ((1.0 - alpha) * self.H + alpha * target_H).astype(np.float32)
            self.T = ((1.0 - alpha) * self.T + alpha * target_T).astype(np.float32)
        else:
            self.H = target_H.astype(np.float32)
            self.T = target_T.astype(np.float32)
            self.has_previous_frame = True


    def sweep(self):
        accepted = 0

        for parity in (0, 1):
            random_values = self.rng.random((self.N, self.N))
            accepted += _update_checkerboard(
                self.spins, self.H, self.T, random_values, parity, self.J
            )

        return accepted / (self.N * self.N)


    def update_surfaces(self):
        if self.temperature_surface is not None:
            self.temperature_surface.remove()
        if self.field_surface is not None:
            self.field_surface.remove()

        step = self.surface_step
        T_surface = self.T[::step, ::step]
        H_surface = self.H[::step, ::step]

        self.temperature_surface = self.ax_temperature.plot_surface(
            self.X_surface, self.Y_surface, T_surface, cmap="inferno", vmin=self.T_bulk, vmax=self.T_edge, linewidth=0,
            antialiased=False, rcount=T_surface.shape[0], ccount=T_surface.shape[1])

        self.field_surface = self.ax_field.plot_surface(
            self.X_surface, self.Y_surface, H_surface, cmap="coolwarm", vmin=-self.H_strength, vmax=self.H_strength,
            linewidth=0, antialiased=False, rcount=H_surface.shape[0], ccount=H_surface.shape[1]) 


    def initialise_plot(self):
        self.fig = plt.figure(figsize=(10, 11))
        grid = self.fig.add_gridspec(2, 2, height_ratios=(1.25, 1.0))

        self.ax_spin = self.fig.add_subplot(grid[0, :])
        self.ax_temperature = self.fig.add_subplot(grid[1, 0], projection="3d")
        self.ax_field = self.fig.add_subplot(grid[1, 1], projection="3d")

        self.spin_image = self.ax_spin.imshow(self.spins, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")

        self.title = self.ax_spin.set_title("Bad Apple Ising Simulation")
        self.ax_spin.set_xticks([])
        self.ax_spin.set_yticks([])

        self.ax_temperature.set_title("Temperature field")
        self.ax_temperature.set_xlabel("x")
        self.ax_temperature.set_ylabel("y")
        self.ax_temperature.set_zlabel("T")
        self.ax_temperature.set_xlim(0, self.N - 1)
        self.ax_temperature.set_ylim(0, self.N - 1)
        self.ax_temperature.set_zlim(self.T_bulk - 0.1, self.T_edge + 0.1)
        self.ax_temperature.view_init(elev=35, azim=-60)

        self.ax_field.set_title("Magnetic field")
        self.ax_field.set_xlabel("x")
        self.ax_field.set_ylabel("y")
        self.ax_field.set_zlabel("H")
        self.ax_field.set_xlim(0, self.N - 1)
        self.ax_field.set_ylim(0, self.N - 1)
        self.ax_field.set_zlim(-1.05 * self.H_strength, 1.05 * self.H_strength)
        self.ax_field.view_init(elev=35, azim=-60)

        self.update_surfaces()
        self.fig.tight_layout()


    def initialise_animation(self):
        return (self.spin_image, self.title, self.temperature_surface, self.field_surface)


    def update(self, frame_number):
        image = self.read_frame()
        if image is None:
            if self.animation is not None:
                self.animation.event_source.stop()
            return self.spin_image, self.title

        is_initial_frame = not self.has_previous_frame
        self.construct_fields(image)

        sweep_count = self.initial_sweeps if is_initial_frame else self.sweeps_per_frame
        acceptance_values = [self.sweep() for _ in range(sweep_count)]
        acceptance = np.mean(acceptance_values)

        self.spin_image.set_data(self.spins)
        self.title.set_text(f"Bad Apple Ising simulation | frame {frame_number + 1}/{self.frame_count} | "
                            f"m = {self.spins.mean():+.3f} | acceptance = {acceptance:.3f}")

        self.update_surfaces()

        return (self.spin_image, self.title, self.temperature_surface, self.field_surface)


    def run(self, output_path: str | None = None, output_fps: float | None = None, dpi: int = 120):
        if output_path is not None:
            output_file = Path(output_path).resolve()
            if output_file == self.video_path.resolve():
                raise ValueError("output_path must not overwrite the input video.")

        self.initialise_plot()

        fps = self.input_fps if output_fps is None else output_fps
        self.animation = FuncAnimation(self.fig, self.update, frames=self.frame_count, init_func=self.initialise_animation,
                                       interval=1000.0 / fps, blit=False, repeat=False, cache_frame_data=False)

        try:
            if output_path is not None:
                writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=6000, extra_args=["-pix_fmt", "yuv420p"])
                with tqdm(total=self.frame_count, desc="Encoding video", unit="frame") as progress:
                    try:
                        self.animation.save(output_path, writer=writer, dpi=dpi, 
                                            progress_callback=lambda _frame, _total: progress.update(1)
                                            )
                    except KeyboardInterrupt:
                        print(f"\nSaved the frames completed so far to {output_file}.")
        finally:
            self.cap.release()

        return self.animation


if __name__ == "__main__":
    simulation = BadAppleIsing(
        video_path="bad-apple.mp4", N=500, J=1.0, H_strength=1.25, distance_scale=10.0, T_bulk=1.5, T_edge=2.55,
        edge_width=10.0, spatial_blur=1.5, temporal_smoothing=0.18, field_spatial_smoothing=0.75,
        sweeps_per_frame=10, invert_video=False, seed=7)

    animation = simulation.run(output_path="big_bad_apple_ising.mp4", output_fps=24, dpi=120)


# ffmpeg -i "ising_bad_apple_full.mp4" -i "bad-apple.mp4" 
# -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -strict -2 -b:a 192k -shortest 
# "ising_bad_apple_with_audio.mp4"