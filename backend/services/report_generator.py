"""
PDF Report Generator for Smart Lab Chair Monitoring System.
Generates professional PDF reports with analysis results.
"""

import os
from datetime import datetime
from typing import Dict, Any
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# Colors
PRIMARY = HexColor("#6c5ce7")
SUCCESS = HexColor("#00b894")
DANGER = HexColor("#d63031")
WARNING = HexColor("#fdcb6e")
DARK = HexColor("#2d3436")
LIGHT_BG = HexColor("#f5f6fa")
MEDIUM_GRAY = HexColor("#636e72")


def generate_pdf_report(
    analysis: Dict[str, Any],
    output_path: str,
    upload_image_path: str = None,
    result_image_path: str = None
) -> str:
    """
    Generate a comprehensive PDF report for a chair arrangement analysis.
    
    Args:
        analysis: Analysis data dictionary from database
        output_path: Where to save the PDF
        upload_image_path: Path to original uploaded image
        result_image_path: Path to annotated result image
    
    Returns:
        Path to the generated PDF file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=PRIMARY,
        spaceAfter=10,
        alignment=TA_CENTER
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=MEDIUM_GRAY,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=DARK,
        spaceBefore=15,
        spaceAfter=10,
        borderPadding=5
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=DARK,
        spaceAfter=6
    )

    elements = []

    # --- Title ---
    elements.append(Paragraph("🪑 Smart Lab Chair Monitoring", title_style))
    elements.append(Paragraph("AI-Powered Arrangement Analysis Report", subtitle_style))
    elements.append(Spacer(1, 5 * mm))

    # --- Metadata ---
    created = analysis.get("created_at", "N/A")
    completed = analysis.get("completed_at", "N/A")
    lab_room = analysis.get("lab_room", "N/A")
    uploaded_by = analysis.get("uploaded_by", "N/A")
    filename = analysis.get("original_filename", "N/A")

    meta_data = [
        ["Report ID", analysis.get("id", "N/A")[:12] + "..."],
        ["Lab Room", lab_room],
        ["Uploaded By", uploaded_by],
        ["Original File", filename],
        ["Upload Time", created],
        ["Analysis Time", completed],
    ]

    meta_table = Table(meta_data, colWidths=[50 * mm, 120 * mm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_BG),
        ('TEXTCOLOR', (0, 0), (0, -1), DARK),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, MEDIUM_GRAY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 8 * mm))

    # --- Summary ---
    elements.append(Paragraph("📊 Analysis Summary", heading_style))

    total_chairs = analysis.get("total_chairs", 0)
    total_desks = analysis.get("total_desks", 0)
    correct = analysis.get("correct_chairs", 0)
    misplaced = analysis.get("misplaced_chairs", 0)
    accuracy = analysis.get("accuracy", 0)
    avg_conf = analysis.get("avg_confidence", 0)

    status_text = "✅ All Chairs Properly Arranged" if misplaced == 0 else "⚠️ Misplaced Chairs Detected"

    summary_data = [
        ["Metric", "Value"],
        ["Detection Status", status_text],
        ["Total Chairs Detected", str(total_chairs)],
        ["Total Desks Detected", str(total_desks)],
        ["Properly Arranged", f"{correct} ✅"],
        ["Misplaced", f"{misplaced} ❌"],
        ["Arrangement Accuracy", f"{accuracy}%"],
        ["Average Confidence", f"{avg_conf}%"],
    ]

    summary_table = Table(summary_data, colWidths=[80 * mm, 90 * mm])
    
    # Header row style
    header_style_list = [
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, MEDIUM_GRAY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
    ]
    
    # Color the status row based on result
    if misplaced > 0:
        header_style_list.append(('TEXTCOLOR', (1, 1), (1, 1), DANGER))
    else:
        header_style_list.append(('TEXTCOLOR', (1, 1), (1, 1), SUCCESS))
    
    summary_table.setStyle(TableStyle(header_style_list))
    elements.append(summary_table)
    elements.append(Spacer(1, 8 * mm))

    # --- Original Image ---
    if upload_image_path and os.path.exists(upload_image_path):
        elements.append(Paragraph("📷 Original Uploaded Image", heading_style))
        try:
            img = RLImage(upload_image_path)
            img_width = min(170 * mm, img.drawWidth)
            scale = img_width / img.drawWidth
            img.drawWidth = img_width
            img.drawHeight = img.drawHeight * scale
            if img.drawHeight > 120 * mm:
                scale2 = 120 * mm / img.drawHeight
                img.drawHeight = 120 * mm
                img.drawWidth = img.drawWidth * scale2
            elements.append(img)
        except Exception:
            elements.append(Paragraph("(Image could not be embedded)", body_style))
        elements.append(Spacer(1, 8 * mm))

    # --- Annotated Result Image ---
    if result_image_path and os.path.exists(result_image_path):
        elements.append(Paragraph("🔍 ML Model Analysis Result", heading_style))
        try:
            img = RLImage(result_image_path)
            img_width = min(170 * mm, img.drawWidth)
            scale = img_width / img.drawWidth
            img.drawWidth = img_width
            img.drawHeight = img.drawHeight * scale
            if img.drawHeight > 120 * mm:
                scale2 = 120 * mm / img.drawHeight
                img.drawHeight = 120 * mm
                img.drawWidth = img.drawWidth * scale2
            elements.append(img)
        except Exception:
            elements.append(Paragraph("(Image could not be embedded)", body_style))
        elements.append(Spacer(1, 8 * mm))

    # --- Individual Chair Details ---
    details = analysis.get("details")
    if details and details.get("chairs"):
        elements.append(PageBreak())
        elements.append(Paragraph("📋 Individual Chair Details", heading_style))

        chair_header = ["Chair #", "Status", "Confidence", "Alignment", "Issues"]
        chair_rows = [chair_header]

        for chair in details["chairs"]:
            cid = chair.get("chair_id", 0)
            is_ok = chair.get("is_properly_arranged", False)
            conf = chair.get("confidence", 0)
            score = chair.get("alignment_score", 0)
            issues = chair.get("issues", [])

            status = "✅ OK" if is_ok else "❌ Misplaced"
            issues_text = "; ".join(issues) if issues else "None"

            chair_rows.append([
                str(cid),
                status,
                f"{conf:.0%}",
                f"{score:.0f}%",
                issues_text[:50]
            ])

        chair_table = Table(chair_rows, colWidths=[18 * mm, 30 * mm, 25 * mm, 25 * mm, 72 * mm])
        chair_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, MEDIUM_GRAY),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
        ]))
        elements.append(chair_table)

    # --- Footer ---
    elements.append(Spacer(1, 15 * mm))
    footer_style = ParagraphStyle(
        'Footer', parent=styles['Normal'],
        fontSize=8, textColor=MEDIUM_GRAY, alignment=TA_CENTER
    )
    elements.append(Paragraph(
        f"Generated by Smart Lab Chair Monitoring System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        footer_style
    ))
    elements.append(Paragraph(
        "Powered by YOLOv8 AI • OpenCV • FastAPI",
        footer_style
    ))

    # Build PDF
    doc.build(elements)
    return output_path
