"""Create the separate two-page approach summary requested by Unstop."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "Vulcans_approach_summary.pdf"
NAVY = colors.HexColor("#17365B")
BLUE = colors.HexColor("#276CA6")
PALE = colors.HexColor("#EAF2F8")
GRAY = colors.HexColor("#536477")


def footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#D4DEE8"))
    canvas.line(0.58 * inch, 0.52 * inch, width - 0.58 * inch, 0.52 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(0.58 * inch, 0.36 * inch, "Vulcans  |  Amazon ML Challenge 2026")
    canvas.drawRightString(width - 0.58 * inch, 0.36 * inch, f"Page {doc.page} of 2")
    canvas.restoreState()


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=17, leading=21, textColor=NAVY, alignment=TA_LEFT, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="SubtitleCustom", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9.2, leading=13, textColor=GRAY, spaceAfter=11,
    ))
    styles.add(ParagraphStyle(
        name="HeadingCustom", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=10.8, leading=13.5, textColor=BLUE, spaceBefore=10,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.2, leading=13.1, textColor=colors.HexColor("#223244"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="BulletCustom", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.2, leading=13.1, leftIndent=13, firstLineIndent=-9,
        textColor=colors.HexColor("#223244"), spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="TableHeadCustom", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8.7, leading=11, textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="TableCellCustom", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8.7, leading=11.5, textColor=colors.HexColor("#223244"),
    ))
    P = lambda text: Paragraph(text, styles["BodyCustom"])
    H = lambda text: Paragraph(text, styles["HeadingCustom"])
    B = lambda text: Paragraph("&#8226;&nbsp;&nbsp;" + text, styles["BulletCustom"])

    story = [
        Paragraph("Business Entity Resolution", styles["TitleCustom"]),
        Paragraph("Amazon ML Challenge 2026  |  Team Vulcans  |  27 September 2026",
                  styles["SubtitleCustom"]),
        HRFlowable(width="100%", thickness=1.5, color=NAVY),
        H("Problem and design goal"),
        P("For each deduplicated Source 1 business, predict all matching Source 2 and Source 3 "
          "IDs, including an empty list for singletons. Training has 2.21 million Source 1 records, "
          "10.32 million targets, and 7.64 million labeled links. Test adds France with no French "
          "training labels. The metric is macro F0.5 over Source 1 groups, so false merges and "
          "missed singletons matter alongside retrieval recall."),
        H("1. Retrieve a bounded, complementary candidate set"),
        P("A local DuckDB index stores Unicode name and address tokens, target frequencies, and "
          "compact legal-suffix name keys. Each query uses rare name and address token postings "
          "within the same country. Separate name and address Jaro-Winkler top-100 quotas protect "
          "matches when either field is corrupted or absent. Compact and accent-folded name "
          "equality rescue punctuation and French accents. For India, an additional top-50 "
          "address-token-overlap channel helps cross-script names. Country is an open-set string, "
          "so France is processed without hard-coded exclusion."),
        H("2. Score pairs with a locally trained model"),
        P("A LightGBM classifier trained on 3.95 million retrieved pairs from a seeded 20,000-query "
          "training sample uses 27 features: Unicode-normalized name and address similarities, "
          "token overlap, source/missingness signals, and four soft first-address-number features. "
          "It uses only supplied data; no external business lookup or remote inference. The "
          "score threshold is 0.75, with a development-selected India override of 0.65."),
        H("3. Resolve group and ownership conflicts"),
        P("Rank selected links by model probability and retain at most 11 per Source 1, the largest "
          "true group in the full training labels. Assign a target predicted by multiple Source 1 "
          "records to its highest-scoring owner. Both passes preserve the exact candidate subset "
          "contract. A fixed address-number veto was rejected for this model because it reduced "
          "labeled F0.5."),
        H("Retrieval checks that changed the design"),
        B("The core candidate route reached 91.18% link recall on all 22,133 development "
          "queries with 4.52 million pairs; its oracle macro F0.5 ceiling was 0.9617."),
        B("India address-overlap top-50 added 686 true links for 171,364 extra pairs and "
          "raised cross-script recall from 62.53% to 70.76% versus the core route."),
        B("Accent-folded core-name equality independently added 328 true links for 53,138 "
          "extra pairs. These two rescue routes were measured separately before union."),
        PageBreak(),
        Paragraph("Measured evidence and submission", styles["TitleCustom"]),
        Paragraph("All scores below are local macro F0.5; public and private leaderboard scores "
                  "must be measured separately.", styles["SubtitleCustom"]),
        HRFlowable(width="100%", thickness=1.5, color=NAVY),
        H("Held-out evaluation"),
    ]
    table_data = [
        [Paragraph("Check", styles["TableHeadCustom"]),
         Paragraph("Scope", styles["TableHeadCustom"]),
         Paragraph("Result", styles["TableHeadCustom"])],
        ["Combined candidate oracle", "22,133 development queries", "0.96793 upper bound"],
        ["Original 23-feature baseline", "Full development, 3 output passes", "0.88331"],
        ["27-feature model, uniform 0.75", "Full development, cap + owner", "0.89574"],
        ["27-feature model, India 0.65", "Full development, cap + owner", "0.89761"],
        ["27-feature model, India 0.65", "Frozen 2,000-query validation", "0.89272"],
    ]
    table = Table(table_data, colWidths=[2.08 * inch, 2.32 * inch, 1.48 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD7E3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [table,
        H("What the measurements mean"),
        B("The candidate oracle assumes perfect scoring on retrieved pairs. It measures the "
          "retrieval ceiling, not an achieved model score."),
        B("The India override improved frozen validation from 0.89091 to 0.89272 and full "
          "development from 0.89574 to 0.89761; it also added false links and reduced "
          "singleton accuracy. The challenge metric favored it on both checks."),
        B("France has no truth labels. In a 20,000-query test slice, a generic-name collision "
          "tail remained after scoring, so the top-11 cap is retained. No French F0.5 is claimed."),
        H("Reproducibility and fair play"),
        P("The final ZIP contains the exact last-stage candidate TSV, matching TSV, runnable "
          "Python source, both locally trained model weights required by the pipeline, pinned "
          "dependencies, and the full methodology. The candidate TSV lists every pair fed to "
          "the final scoring model, and every predicted ID is a member of that row's candidate "
          "set. Streaming and official validators are run before submission. No supplied record "
          "text is sent to a remote service."),
        H("Team"),
        P("Yash Kailas Doke, Harsh Jitendra Jain, Ayush Tiwari, and Vedant Kaulgekar."),
    ]
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=0.58 * inch,
        rightMargin=0.58 * inch, topMargin=0.62 * inch,
        bottomMargin=0.67 * inch, title="Vulcans - Amazon ML Challenge 2026 Approach",
        author="Team Vulcans",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
