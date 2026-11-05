import os
import time
from playwright.sync_api import sync_playwright
from PIL import Image

OUTPUT_DIR = "docs/assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

URL = "https://transient-finder.streamlit.app/~/+/"

def capture_demo():
    print(f"Connecting to live Streamlit instance: {URL}...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # 1280x780 viewport gives standard aspect ratio for GitHub readmes
        context = browser.new_context(
            viewport={"width": 1280, "height": 780},
            device_scale_factor=1.5
        )
        page = context.new_page()

        page.goto(URL, timeout=60000, wait_until="domcontentloaded")
        print("Page opened, waiting for Streamlit container to hydrate...")
        page.wait_for_selector('[data-testid="stApp"]', timeout=45000)
        time.sleep(6)  # Allow live data ingestion & Plotly 3D charts to settle

        def scroll_main(top_px):
            page.locator('[data-testid="stMain"]').evaluate(f'el => el.scrollTop = {top_px}')
            time.sleep(1.2)

        def click_radio(label_text):
            radio = page.locator('[data-testid="stRadio"] label').filter(has_text=label_text)
            radio.click()
            time.sleep(2.5)

        raw_frames = []

        def snap(name, duration_ms=2200):
            png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
            page.screenshot(path=png_path)
            print(f"Captured screenshot: {png_path}")
            im = Image.open(png_path).convert("RGB")
            raw_frames.append((im, duration_ms))

        # --- Scene 1: Live Stream Top & Candidate Table ---
        print("Scene 1: Live Survey Stream Table...")
        scroll_main(0)
        snap("dashboard_live_stream", duration_ms=2500)

        # --- Scene 2: Candidate Deep-Dive & 3D Heliocentric Orbit ---
        print("Scene 2: Candidate AI Deep-Dive & 3D Orbit...")
        scroll_main(680)
        snap("orbit_3d_diagnostics", duration_ms=2800)

        # --- Scene 3: Heuristics & Diagnostics Breakdown ---
        print("Scene 3: Heuristic Pillars & Diagnostics...")
        scroll_main(1180)
        snap("heuristics_breakdown", duration_ms=2200)

        # --- Scene 4: 6-Month Re-Validation Lab ---
        print("Scene 4: 6-Month Re-Validation Lab...")
        click_radio("6-Month Re-Validation Lab")
        scroll_main(0)
        snap("revalidation_lab", duration_ms=2600)

        # --- Scene 5: Ground-Truth Benchmarks ---
        print("Scene 5: Ground-Truth & False-Positive Benchmarks...")
        click_radio("Ground-Truth")
        scroll_main(0)
        snap("benchmark_metrics", duration_ms=2400)
        scroll_main(500)
        snap("benchmark_confusion_matrix", duration_ms=2400)

        # --- Scene 6: Model Insights & Feature Ranking ---
        print("Scene 6: Model Insights & Feature Importance...")
        click_radio("Model Insights")
        scroll_main(0)
        snap("model_insights", duration_ms=2400)

        browser.close()

        # --- Compile into optimized Animated GIF ---
        print(f"Compiling {len(raw_frames)} frames into animated GIF...")
        target_width = 960
        processed_frames = []
        durations = []

        for im, dur in raw_frames:
            target_height = int(target_width * im.height / im.width)
            resized = im.resize((target_width, target_height), Image.Resampling.LANCZOS)
            # Quantize to adaptive web palette for crisp clarity and small file size
            quantized = resized.convert("P", palette=Image.ADAPTIVE, colors=128)
            processed_frames.append(quantized)
            durations.append(dur)

        gif_path = os.path.join(OUTPUT_DIR, "live_demo.gif")
        processed_frames[0].save(
            gif_path,
            save_all=True,
            append_images=processed_frames[1:],
            optimize=True,
            duration=durations,
            loop=0
        )

        size_mb = os.path.getsize(gif_path) / (1024 * 1024)
        print(f"Successfully generated animated demo GIF: {gif_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    capture_demo()
