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
        H("1. Retrieve candidates with complementary routes"),
        P("A local DuckDB index stores Unicode name and address tokens, target frequencies, and "
          "compact legal-suffix name keys. Each query uses rare name and address token postings "
          "within the same country. Separate name and address Jaro-Winkler top-100 quotas protect "
          "matches when either field is corrupted or absent. Compact and accent-folded name "
          "equality rescue punctuation and French accents. For India, an additional top-50 "
          "address-token-overlap channel helps cross-script names. Country is an open-set string, "
          "so France is processed without hard-coded exclusion."),
        H("2. Learned last-stage block, then final matching"),
        P("A frozen 23-feature LightGBM ranks the broad union and passes at most 40 candidates "
          "per Source 1. A local word TF-IDF search adds up to 20 IDs per India query. For US "
          "queries with at most three initial matches, a second exact sparse TF-IDF address "
          "search adds up to five IDs. The recorded final candidate TSV is the union actually "
          "scored. Top 40 retained 70,735/70,741 broad-reachable true links in development; "
          "the India quota added 1,346 reachable links. No pretrained weights or external "
          "data are used for these searches."),
        P("A separate 35-feature Unicode/transliteration LightGBM was trained on 5.25 million "
          "pairs mined from 190,086 training-owned queries. A 36-feature specialist replaces "
          "its score for targets with empty addresses, using target core-name frequency. "
          "India/other thresholds are 0.65/0.75; blank-address threshold is 0.80."),
        H("3. Resolve group and ownership conflicts"),
        P("Rank selected links by model probability and retain at most 11 per Source 1, the largest "
          "true group in the full training labels. Assign a target predicted by multiple Source 1 "
          "records to its highest-scoring owner. Both passes preserve the exact candidate subset "
          "contract. A fixed address-number veto was rejected for this model because it reduced "
          "labeled F0.5."),
        H("Retrieval checks that changed the design"),
        B("The broad lexical union had 4.74 million development pairs and an oracle macro "
          "F0.5 ceiling of 0.96793. The learned top-40 block kept the ceiling at 0.96788."),
        B("A full-index India address TF-IDF search found 1,870/3,070 missed true links "
          "among its top 100 address keys. Its bounded quota improved measured end-to-end "
          "F0.5 on development and separate frozen validation."),
        B("The gated US top-five address route added 28 true links and zero false links on "
          "frozen validation; the same final matcher scored every added pair."),
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
        ["Learned top-40 oracle", "22,133 development queries", "0.96788 upper bound"],
        ["Original 23-feature baseline", "Full development, 3 output passes", "0.88331"],
        ["Hard + blank specialist", "Development, top 40, cap + owner", "0.92322"],
        ["Plus India address top 20", "Development, cap + owner", "0.93183"],
        ["Hard + blank specialist", "Frozen 1,994-query validation", "0.91845"],
        ["Plus India address top 20", "Frozen 1,994-query validation", "0.92752"],
        ["Plus gated US address top 5", "Frozen 1,994-query validation", "0.92989"],
    ]
    table = Table(table_data, colWidths=[2.08 * inch, 2.32 * inch, 1.48 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD7E3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [table,
        H("What the measurements mean"),
        B("The candidate oracle assumes perfect scoring on retrieved pairs. It measures the "
          "retrieval ceiling, not an achieved model score."),
        B("The learned top-40 block controls most candidate volume. India address top 20 "
          "raised frozen macro F0.5 by 0.00907; gated US address top 5 added 0.00237. "
          "Both quotas were measured before full test inference."),
        B("The general matcher compares original Unicode and local ASCII-transliterated "
          "name/address views, plus soft number agreement. The blank-address specialist "
          "uses locally computed core-name rarity. No French labels guided either model."),
        B("France has no truth labels. In a 20,000-query test slice, generic-name collisions "
          "remained before capping. The top-11 cap was retained; no French F0.5 is claimed."),
        H("Reproducibility and fair play"),
        P("The final ZIP contains the exact last-stage candidate TSV, matching TSV, runnable "
          "Python source, four locally trained model weights, pinned dependencies, and the full "
          "methodology. Both broad lexical and TF-IDF indexes are reproducible intermediates "
          "from supplied records. Every predicted ID belongs to the submitted candidate set. "
          "Streaming and official validators are run before submission. No supplied record "
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
