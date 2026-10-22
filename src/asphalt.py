#!/usr/bin/env python3
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data


# --- Noise & Texture functions (same as before) ---
def value_noise_2d(shape, cell, rng):
    H, W = shape
    cell = max(int(cell), 2)
    gh, gw = H // cell + 2, W // cell + 2
    grid = rng.random((gh, gw))
    ys = np.linspace(0, gh - 2, H, endpoint=False)
    xs = np.linspace(0, gw - 2, W, endpoint=False)
    y0, x0 = ys.astype(int), xs.astype(int)
    y1, x1 = y0 + 1, x0 + 1
    fy = (ys - y0)[:, None]
    fy = fy * fy * (3 - 2 * fy)
    fx = (xs - x0)[None, :]
    fx = fx * fx * (3 - 2 * fx)
    v00 = grid[y0[:, None], x0[None, :]]
    v01 = grid[y0[:, None], x1[None, :]]
    v10 = grid[y1[:, None], x0[None, :]]
    v11 = grid[y1[:, None], x1[None, :]]
    top = v00 + fx * (v01 - v00)
    bot = v10 + fx * (v11 - v10)
    return top + fy * (bot - top)


def fbm(shape, rng, octaves=6, base=64, gain=0.5, lac=2.0):
    out = np.zeros(shape)
    amp, cell, norm = 1.0, float(base), 0.0
    for _ in range(octaves):
        out += amp * value_noise_2d(shape, cell, rng)
        norm += amp
        amp *= gain
        cell = max(cell / lac, 2.0)
    return out / norm


def make_cracks(shape, n, rng, steps=600, step_len=1.5, ang_noise=0.25):
    """Now uses a soft brush stroke so cracks aren't jagged pixels."""
    H, W = shape
    mask = np.zeros(shape, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]

    for _ in range(n):
        y, x = rng.uniform(0, H), rng.uniform(0, W)
        a = rng.uniform(0, 2 * np.pi)
        for _ in range(rng.integers(steps // 2, steps)):
            a += rng.normal(0, ang_noise)
            y += step_len * np.sin(a)
            x += step_len * np.cos(a)

            # Soft brush stroke (radius ~2 pixels)
            y0, y1 = max(0, int(y) - 4), min(H, int(y) + 5)
            x0, x1 = max(0, int(x) - 4), min(W, int(x) + 5)
            if y0 >= y1 or x0 >= x1:
                continue

            dist = np.sqrt((yy[y0:y1, x0:x1] - y) ** 2 + (xx[y0:y1, x0:x1] - x) ** 2)
            # Exponentially decaying brush
            mask[y0:y1, x0:x1] = np.maximum(
                mask[y0:y1, x0:x1], np.exp(-(dist**2) / 2.0)
            )
    return mask


def save_ppm(path, img):
    H, W = img.shape[:2]
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (W, H))
        f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())


def build_texture(crack_mask, rng):
    H, W = crack_mask.shape
    base = np.array([0.25, 0.25, 0.27], np.float32)  # Darker asphalt
    tex = np.broadcast_to(base, (H, W, 3)).copy()
    tex += rng.normal(0, 0.03, (H, W, 3)).astype(np.float32)

    # Aggregate
    n = (H * W) // 40
    ys, xs = rng.integers(0, H, n), rng.integers(0, W, n)
    delta = rng.uniform(-0.15, 0.15, (n, 3)).astype(np.float32)
    np.add.at(tex, (ys, xs), delta)

    # Dark cracks
    m = crack_mask[..., None]
    crack_rgb = np.array([0.02, 0.02, 0.03], np.float32)
    tex = tex * (1 - 0.95 * m) + crack_rgb * 0.95 * m
    return np.clip(tex, 0, 1)


# --- Main ---
def main():
    headless = "--headless" in sys.argv
    p.connect(p.DIRECT if headless else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)

    N, SIZE, seed = 256, 20.0, 42
    rng = np.random.default_rng(seed)

    # Heightfield
    h = 0.02 * (fbm((N, N), rng, octaves=4, base=128) - 0.5)
    h += 0.002 * (fbm((N, N), rng, octaves=5, base=8) - 0.5)
    cracks = make_cracks((N, N), n=20, rng=rng)

    # Sharper depression for the cracks
    h -= 0.02 * np.clip(cracks * 3.0, 0, 1.0)

    col = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[SIZE / N, SIZE / N, 1.0],
        heightfieldData=h.flatten(order="F").tolist(),
        numHeightfieldRows=N,
        numHeightfieldColumns=N,
        heightfieldTextureScaling=(N - 1) / 2,
    )
    ground = p.createMultiBody(0, col)

    # Physics
    p.changeDynamics(
        ground, -1, lateralFriction=0.9, rollingFriction=0.01, spinningFriction=0.01
    )

    # Texture & Material
    tex = build_texture(cracks, rng)
    save_ppm("asphalt_tex.ppm", (tex * 255).astype(np.uint8))
    tid = p.loadTexture("asphalt_tex.ppm")

    # FORCE the GUI to be matte and dark gray instead of shiny white
    # This fixes the OpenGL3 heightfield material issue
    p.changeVisualShape(
        ground,
        -1,
        rgbaColor=[0.35, 0.35, 0.35, 1.0],  # Dark gray
        specularColor=[0.05, 0.05, 0.05],  # Kill the shine
        textureUniqueId=tid if tid >= 0 else -1,
    )

    # Drop a ball
    cb = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
    vb = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=[1.0, 0.25, 0.25, 1])
    p.createMultiBody(0.8, cb, vb, basePosition=[0, 0, 1.5])

    # Camera
    p.resetDebugVisualizerCamera(
        cameraDistance=14, cameraYaw=45, cameraPitch=-40, cameraTargetPosition=[0, 0, 0]
    )

    # Top-down render
    vm = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0],
        distance=10,
        yaw=0,
        pitch=-89,
        roll=0,
        upAxisIndex=2,
    )
    pm = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100)
    _, _, rgb, _, _ = p.getCameraImage(512, 512, vm, pm)
    rgb = np.array(rgb, dtype=np.uint8).reshape((512, 512, 4))[..., :3]
    save_ppm("topdown.ppm", rgb)
    print("Saved topdown.ppm")

    # Run
    if not headless:
        t0 = time.time()
        try:
            while p.isConnected() and (time.time() - t0 < 60):
                p.stepSimulation()
                time.sleep(1 / 240)
        except p.error, KeyboardInterrupt:
            pass

    p.disconnect()


if __name__ == "__main__":
    main()
