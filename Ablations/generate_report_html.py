import os
from pathlib import Path

def generate_midog_master_index(base_reports_dir="../ablation_reports", output_file="index.html"):
    base_path = Path(base_reports_dir)
    
    # Define the tracks we expect
    tracks = {
        "Track 1: Mitotic Figure Detection": base_path / "Track1",
        "Track 2: Atypical Mitosis Classification": base_path / "Track2"
    }

    sections_html = ""

    for track_name, track_path in tracks.items():
        if not track_path.exists():
            print(f"Warning: {track_path} not found. Skipping {track_name}.")
            continue

        report_files = sorted([f for f in track_path.glob("*.html")])
        
        if not report_files:
            continue

        # Start a section for this track
        sections_html += f"""
        <section class="track-section">
            <h2 class="track-title">{track_name}</h2>
            <div class="grid">
        """

        for report in report_files:
            # Clean up the filename for display (e.g., ablation_report_user.html -> user)
            display_name = report.stem.replace("ablation_report_", "")
            
            # Construct a relative path for the link
            # This assumes index.html is in the parent of the Track folders
            rel_link = f"{track_path.name}/{report.name}"

            sections_html += f"""
                <div class="card">
                    <div class="card-body">
                        <h3>{display_name}</h3>
                        <p class="meta">Container Ablation Report</p>
                        <a href="{rel_link}" class="btn">View Diffs</a>
                    </div>
                </div>
            """
        
        sections_html += "</div></section>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MIDOG 2025 | Ablation Study Transparency</title>
        <style>
            :root {{
                --midog-purple: rgb(39, 3, 65);
                --midog-gold: #f39c12; /* Accent color for buttons/highlights */
                --bg-light: #fdfdfd;
                --text-dark: #2c3e50;
                --border-muted: #e1e4e8;
            }}
            body {{
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                margin: 0; padding: 0;
                background-color: var(--bg-light);
                color: var(--text-dark);
                line-height: 1.6;
            }}
            header {{
                background-color: var(--midog-purple);
                color: white;
                padding: 80px 20px;
                text-align: center;
                box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            }}
            header h1 {{ margin: 0; font-size: 2.5rem; letter-spacing: 1px; }}
            header p {{ opacity: 0.9; font-weight: 300; font-size: 1.1rem; }}
            
            .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 20px; }}
            
            .intro-card {{
                background: white;
                border-radius: 8px;
                padding: 30px;
                margin-bottom: 50px;
                box-shadow: 0 2px 15px rgba(0,0,0,0.05);
                border-top: 5px solid var(--midog-purple);
            }}
            
            .track-section {{ margin-bottom: 60px; }}
            .track-title {{ 
                font-size: 1.8rem; 
                color: var(--midog-purple); 
                border-bottom: 2px solid var(--midog-purple);
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
                gap: 25px;
            }}
            
            .card {{
                background: white;
                border: 1px solid var(--border-muted);
                border-radius: 10px;
                transition: all 0.3s ease;
                display: flex;
                flex-direction: column;
            }}
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(39, 3, 65, 0.15);
                border-color: var(--midog-purple);
            }}
            .card-body {{ padding: 25px; text-align: center; }}
            .card-body h3 {{ margin: 0 0 10px 0; font-size: 1.2rem; color: #333; }}
            .meta {{ font-size: 0.85rem; color: #666; margin-bottom: 20px; }}
            
            .btn {{
                display: inline-block;
                padding: 10px 20px;
                background-color: var(--midog-purple);
                color: white;
                text-decoration: none;
                border-radius: 25px;
                font-size: 0.9rem;
                font-weight: 600;
                transition: background 0.2s;
            }}
            .btn:hover {{ background-color: #51107d; }}
            
            footer {{
                background: #eee;
                padding: 40px;
                text-align: center;
                font-size: 0.9rem;
                color: #555;
            }}
        </style>
    </head>
    <body>

    <header>
        <h1>MIDOG 2025</h1>
        <p>Inference Script Ablation & Transparency Reports</p>
    </header>

    <div class="container">
        <div class="intro-card">
            <h2>Scientific Transparency</h2>
            <p>
                In accordance with the <strong>MIDOG 2025</strong> commitment to reproducible science, 
                this page documents the systematic modifications applied to participant containers. 
                Our goal is to isolate the performance gains provided by Test-Time Augmentation (TTA) 
                and Model Ensembling across two distinct tracks:
            </p>
            <ul>
                <li><strong>Track 1:</strong> Focuses on the detection of mitotic figures in whole-slide images.</li>
                <li><strong>Track 2:</strong> Targets the classification of atypical mitosis, a clinically significant morphological variant.</li>
            </ul>
        </div>

        {sections_html}
    </div>

    <footer>
        <p>Part of the MIDOG 2025 Challenge Methodology | DeepMicroscopy Group</p>
    </footer>

    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Master Index generated successfully: {output_file}")

if __name__ == "__main__":
    generate_midog_master_index()