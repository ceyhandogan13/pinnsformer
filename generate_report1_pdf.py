"""
Progress Report 1 — PDF generator (ReportLab)
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# ── Output path ───────────────────────────────────────────────────────────────
OUT = 'progress_report_1.pdf'
FIG = 'bs_pinn'

# ── Page template with header/footer ─────────────────────────────────────────
def make_doc():
    doc = SimpleDocTemplate(
        OUT,
        pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.8*cm, bottomMargin=2.5*cm,
        title='Progress Report 1 – Week 1-2',
        author='Ceyhan Doğan'
    )
    return doc

def header_footer(canvas, doc):
    canvas.saveState()
    W, H = A4
    # header line
    canvas.setStrokeColor(colors.HexColor('#2c5f8a'))
    canvas.setLineWidth(0.8)
    canvas.line(2.5*cm, H - 2.0*cm, W - 2.5*cm, H - 2.0*cm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(2.5*cm, H - 1.7*cm, 'CMP712 — Progress Report 1')
    canvas.drawRightString(W - 2.5*cm, H - 1.7*cm, 'Ceyhan Doğan')
    # footer line
    canvas.line(2.5*cm, 1.8*cm, W - 2.5*cm, 1.8*cm)
    canvas.drawCentredString(W / 2, 1.2*cm, str(doc.page))
    canvas.restoreState()

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
BLUE  = colors.HexColor('#1a3d6b')
LBLUE = colors.HexColor('#2c5f8a')
GRAY  = colors.HexColor('#444444')
LGRAY = colors.HexColor('#f0f4f8')

title_style = ParagraphStyle('Title2',
    fontSize=18, fontName='Helvetica-Bold', textColor=BLUE,
    alignment=TA_CENTER, spaceAfter=4)

subtitle_style = ParagraphStyle('Subtitle',
    fontSize=11, fontName='Helvetica', textColor=GRAY,
    alignment=TA_CENTER, spaceAfter=2)

author_style = ParagraphStyle('Author',
    fontSize=11, fontName='Helvetica-Bold', textColor=LBLUE,
    alignment=TA_CENTER, spaceAfter=2)

h1_style = ParagraphStyle('H1',
    fontSize=13, fontName='Helvetica-Bold', textColor=BLUE,
    spaceBefore=14, spaceAfter=6, borderPad=2)

h2_style = ParagraphStyle('H2',
    fontSize=11, fontName='Helvetica-Bold', textColor=LBLUE,
    spaceBefore=10, spaceAfter=4)

body_style = ParagraphStyle('Body2',
    fontSize=10, fontName='Helvetica', leading=15,
    textColor=GRAY, alignment=TA_JUSTIFY, spaceAfter=6)

bullet_style = ParagraphStyle('Bullet',
    fontSize=10, fontName='Helvetica', leading=14,
    textColor=GRAY, leftIndent=16, spaceAfter=3,
    bulletIndent=6)

caption_style = ParagraphStyle('Caption',
    fontSize=8.5, fontName='Helvetica-Oblique', textColor=colors.HexColor('#666666'),
    alignment=TA_CENTER, spaceAfter=8)

math_style = ParagraphStyle('Math',
    fontSize=10, fontName='Helvetica-Oblique', textColor=GRAY,
    alignment=TA_CENTER, spaceBefore=4, spaceAfter=4)

box_style = ParagraphStyle('Box',
    fontSize=10.5, fontName='Helvetica-Bold', textColor=BLUE,
    alignment=TA_CENTER, spaceAfter=2)

# ── Helpers ───────────────────────────────────────────────────────────────────
def H1(text, num):
    return Paragraph(f'{num}. {text}', h1_style)

def H2(text):
    return Paragraph(text, h2_style)

def P(text):
    return Paragraph(text, body_style)

def B(text):
    return Paragraph(f'• {text}', bullet_style)

def Math(text):
    return Paragraph(text, math_style)

def Caption(text):
    return Paragraph(text, caption_style)

def fig(path, width=14*cm, caption_text=''):
    elems = []
    if os.path.exists(path):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            iw, ih = im.size
        height = width * ih / iw
        elems.append(Image(path, width=width, height=height))
    if caption_text:
        elems.append(Caption(caption_text))
    return elems

def section_line():
    return HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#c0cfe0'),
                      spaceAfter=4, spaceBefore=2)

def result_box(rows):
    """Blue-bordered result summary box."""
    data = [[Paragraph(r[0], ParagraphStyle('BL', fontSize=10, fontName='Helvetica-Bold',
                                             textColor=BLUE)),
             Paragraph(r[1], ParagraphStyle('BV', fontSize=10, fontName='Helvetica',
                                             textColor=GRAY))]
            for r in rows]
    t = Table(data, colWidths=[5*cm, 6*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LGRAY),
        ('BOX',        (0, 0), (-1, -1), 1.2, LBLUE),
        ('INNERGRID',  (0, 0), (-1, -1), 0.3, colors.HexColor('#c0cfe0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',  (0, 0), (-1, -1), 8),
    ]))
    return t

def simple_table(headers, rows, col_widths=None):
    header_row = [Paragraph(h, ParagraphStyle('TH', fontSize=9.5,
                             fontName='Helvetica-Bold', textColor=colors.white))
                  for h in headers]
    body_rows = [[Paragraph(str(c), ParagraphStyle('TD', fontSize=9,
                             fontName='Helvetica', textColor=GRAY, leading=13))
                  for c in row] for row in rows]
    data = [header_row] + body_rows
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), LBLUE),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LGRAY]),
        ('BOX',        (0, 0), (-1, -1), 0.8, colors.HexColor('#c0cfe0')),
        ('INNERGRID',  (0, 0), (-1, -1), 0.3, colors.HexColor('#dde5ee')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',  (0, 0), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return t

# ── Build story ───────────────────────────────────────────────────────────────
def build():
    doc  = make_doc()
    story = []

    # ── Title page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.2*cm))
    story.append(Paragraph('CMP712 — Advanced Topics in Machine Learning', subtitle_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph('Progress Report 1 — Week 1–2', title_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width='60%', thickness=2, color=LBLUE,
                             hAlign='CENTER', spaceAfter=10))
    story.append(Paragraph(
        'Transformers in Financial Derivative Pricing:<br/>'
        'Solving the Black-Scholes Equation using PINNsFormer',
        ParagraphStyle('Stitle', fontSize=13, fontName='Helvetica-Bold',
                       textColor=LBLUE, alignment=TA_CENTER, spaceAfter=12, leading=18)))
    story.append(Spacer(1, 0.5*cm))
    story.append(author_style and Paragraph('<b>Ceyhan Doğan</b>', author_style))
    story.append(Paragraph('Hacettepe University — Department of Computer Engineering', subtitle_style))
    story.append(Paragraph('April 2025', subtitle_style))
    story.append(Spacer(1, 0.8*cm))

    # Abstract box
    abs_data = [[Paragraph(
        '<b>Abstract.</b> This report documents Week 1–2 progress of the CMP712 term project, '
        'which adapts PINNsFormer — a Transformer-based Physics-Informed Neural Network — to the '
        'Black-Scholes PDE for European option pricing. We describe the domain transfer from '
        'physical to financial coordinates, synthetic dataset generation, and baseline PINN '
        'training. The baseline achieves <b>rMAE = 0.45%, rRMSE = 0.35%</b> on a 100×100 '
        'evaluation grid after 10 000 epochs, confirming the spectral-bias problem near the '
        'strike price that motivates the PINNsFormer approach.',
        ParagraphStyle('Abs', fontSize=9.5, fontName='Helvetica', leading=14,
                       textColor=GRAY, alignment=TA_JUSTIFY))]]
    abs_t = Table(abs_data, colWidths=[15.5*cm])
    abs_t.setStyle(TableStyle([
        ('BOX', (0,0),(-1,-1), 0.8, LBLUE),
        ('BACKGROUND', (0,0),(-1,-1), LGRAY),
        ('TOPPADDING', (0,0),(-1,-1), 10),
        ('BOTTOMPADDING', (0,0),(-1,-1), 10),
        ('LEFTPADDING', (0,0),(-1,-1), 12),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
    ]))
    story.append(abs_t)
    story.append(PageBreak())

    # ── 1. Problem Definition ─────────────────────────────────────────────────
    story.append(H1('Problem Definition & Motivation', 1))
    story.append(section_line())

    story.append(H2('1.1  The Black-Scholes PDE'))
    story.append(P(
        'The Black-Scholes model governs the price V(S, t) of a European option through:'))
    story.append(Math(
        '∂V/∂t  +  ½ σ² S² ∂²V/∂S²  +  r S ∂V/∂S  −  r V  =  0'))
    story.append(P(
        'where S is the underlying asset price, t is time, r is the risk-free interest rate, '
        'and σ is the volatility. The closed-form solution for a European <b>call</b> option is:'))
    story.append(Math('C(S,τ)  =  S · N(d₁)  −  K e^(−rτ) · N(d₂)'))
    story.append(Math(
        'd₁ = [ln(S/K) + (r + σ²/2)τ] / (σ√τ),      d₂ = d₁ − σ√τ'))
    story.append(P(
        'where K is the strike price, τ = T − t is time-to-maturity, and N(·) is '
        'the standard normal CDF.'))

    story.append(H2('1.2  Motivation for a PINN Approach'))
    story.append(P(
        'Traditional numerical solvers (finite difference, finite element) must be re-run '
        'from scratch for every new parameter combination (K, r, σ). '
        'Physics-Informed Neural Networks (PINNs) embed the PDE residual directly into the '
        'loss function; once trained, the network serves as a <b>mesh-free surrogate</b> '
        'that can be queried at any (S, τ).'))

    story.append(H2('1.3  Motivation for PINNsFormer'))
    story.append(P(
        'Standard PINNs treat each collocation point independently (<i>pointwise</i> training), '
        'causing <b>spectral bias</b> — failure to resolve the kink of the payoff near S = K '
        'and at short maturities τ → 0. PINNsFormer (Zhao et al., 2024) addresses this via:'))
    story.append(B('<b>Pseudo-sequence generator:</b> each point (S, τ) is expanded into a '
                   'temporal sequence [(S, τ), (S, τ+δ), …] fed through a Transformer encoder-decoder.'))
    story.append(B('<b>WaveAct activation:</b> f(x) = w₁ sin(x) + w₂ cos(x) with learnable w₁, w₂ '
                   '— captures high-frequency features that tanh cannot.'))
    story.append(B('<b>Sequential loss:</b> penalises temporal inconsistency across the sequence.'))
    story.append(Spacer(1, 0.2*cm))
    story.append(P('<b>This project</b> is the first to adapt PINNsFormer to the Black-Scholes financial domain.'))

    # ── 2. Literature Review ──────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(H1('Literature Review', 2))
    story.append(section_line())

    lit_headers = ['Reference', 'Contribution', 'Limitation']
    lit_rows = [
        ['Raissi et al. (2019)', 'Foundational PINN framework; PDE embedded in loss', 'Pointwise; spectral bias'],
        ['Dhiman & Hu (2023)',   'PINN for BS call/put; ~30% gain over analytic benchmarks', 'Convergence issues near boundaries'],
        ['Nuugulu et al. (2025)','PINN vs data-driven on BS & Heston; PINN superior in stability', 'Both approaches still pointwise'],
        ['Zhao et al. (2024) ★', 'PINNsFormer: Transformer + WaveAct + sequential loss', 'Not applied to financial PDEs'],
    ]
    story.append(simple_table(lit_headers, lit_rows, col_widths=[3.8*cm, 7.0*cm, 4.7*cm]))
    story.append(Caption('Table 1. Summary of directly relevant prior work. ★ = primary methodological reference.'))

    # ── 3. Domain Transfer ────────────────────────────────────────────────────
    story.append(H1('Domain Transfer: Physical → Financial', 3))
    story.append(section_line())

    story.append(P(
        'PINNsFormer was designed for physics PDEs with spatial input x and time t. '
        'We perform the following mapping:'))

    dt_headers = ['PINNsFormer', 'Black-Scholes', 'Notes']
    dt_rows = [
        ['x  (space)', 'S  (asset price)',          'S ∈ [20, 200]'],
        ['t  (time)',  'τ = T − t  (time-to-maturity)', 'Reversed; τ = 0 is terminal'],
        ['u(x, t)',    'V(S, τ)  (option price)',    ''],
    ]
    story.append(simple_table(dt_headers, dt_rows, col_widths=[4*cm, 6.5*cm, 5*cm]))
    story.append(Caption('Table 2. Domain transfer from PINNsFormer physical coordinates to financial coordinates.'))

    story.append(P(
        'Working with τ instead of t converts the <i>terminal condition</i> '
        'V(S, T) = max(S − K, 0) into an <i>initial condition</i> V(S, 0) = max(S − K, 0), '
        'the natural form for PINNs. The PDE in (S, τ) notation becomes:'))
    story.append(Math('∂V/∂τ  −  ½ σ² S² ∂²V/∂S²  −  r S ∂V/∂S  +  r V  =  0'))

    story.append(H2('3.1  Boundary & Terminal Conditions'))
    bc_headers = ['Condition', 'Expression']
    bc_rows = [
        ['Terminal  (τ = 0)',       'V(S, 0)  =  max(S − K, 0)'],
        ['Left BC  (S → 0)',        'V(S_min, τ)  ≈  0'],
        ['Right BC  (S → ∞)',       'V(S_max, τ)  ≈  S_max − K e^(−rτ)'],
    ]
    story.append(simple_table(bc_headers, bc_rows, col_widths=[5.5*cm, 10*cm]))
    story.append(Caption('Table 3. Boundary and terminal conditions for the European call option.'))

    # ── 4. Dataset ────────────────────────────────────────────────────────────
    story.append(H1('Synthetic Dataset Generation', 4))
    story.append(section_line())

    story.append(H2('4.1  Parameters'))
    story.append(P(
        'All experiments use: K = 100, r = 0.05, σ = 0.20, '
        'S ∈ [20, 200], τ ∈ [0, 1] years.'))

    story.append(H2('4.2  Evaluation Grid'))
    story.append(P(
        'A 100×100 mesh of (S, τ) points is generated. Ground-truth option prices are '
        'computed with the analytical Black-Scholes formula.'))

    story.append(H2('4.3  Put-Call Parity Verification'))
    story.append(P(
        'To validate the dataset, put-call parity C − P = S − K e^(−rτ) was checked. '
        'The maximum absolute error across the τ > 0 region was ≈ 10⁻¹⁴ (machine epsilon), '
        'confirming numerical correctness.'))

    story += fig(f'{FIG}/fig_price_surfaces.png', width=14.5*cm,
                 caption_text='Figure 1. Analytical Black-Scholes price surfaces for the European call (left) '
                              'and put (right). Parameters: K=100, r=0.05, σ=0.20.')

    story.append(H2('4.4  Collocation Points'))
    cp_headers = ['Set', 'Role', 'Size']
    cp_rows = [
        ['ℝ  (Residual)',  'Interior PDE enforcement',      '2 000'],
        ['𝒯  (Terminal)', 'τ = 0 payoff condition',         '500'],
        ['ℬ  (Boundary)', 'S_min and S_max conditions',     '500 per side'],
    ]
    story.append(simple_table(cp_headers, cp_rows, col_widths=[4*cm, 7.5*cm, 4*cm]))
    story.append(Caption('Table 4. Collocation point sets used for PINN training.'))
    story.append(P(
        'Points are sampled uniformly at random with fixed seeds (0, 1, 2) for full reproducibility '
        '(torch.manual_seed(42), np.random.seed(42)).'))

    # ── 5. Baseline PINN ──────────────────────────────────────────────────────
    story.append(H1('Baseline: Standard PINN', 5))
    story.append(section_line())

    story.append(H2('5.1  Architecture'))
    story.append(P(
        'A fully-connected network with 6 hidden layers of width 64, tanh activations, '
        'and a single linear output neuron approximates V(S, τ).'))

    story.append(H2('5.2  Loss Function'))
    story.append(P('The composite PINN loss is:'))
    story.append(Math('ℒ  =  λ₁ ℒ_PDE  +  λ₂ ℒ_TC  +  λ₃ ℒ_BC'))
    story.append(P(
        'where ℒ_PDE is the mean-squared PDE residual over ℝ, ℒ_TC enforces the terminal '
        'payoff condition, and ℒ_BC enforces boundary values. '
        'PDE derivatives are computed via automatic differentiation.'))

    story.append(H2('5.3  Training Setup'))
    ts_headers = ['Hyperparameter', 'Value']
    ts_rows = [
        ['Optimizer',            'Adam'],
        ['Learning rate',        '1 × 10⁻³'],
        ['Epochs',               '10 000'],
        ['λ₁ / λ₂ / λ₃',        '1 / 10 / 10'],
        ['Hidden dimension',     '64'],
        ['Number of layers',     '6'],
        ['Activation',           'tanh'],
        ['Random seed',          '42  (torch + numpy)'],
    ]
    story.append(simple_table(ts_headers, ts_rows, col_widths=[6*cm, 9.5*cm]))
    story.append(Caption('Table 5. PINN training hyperparameters.'))

    # ── 6. Results ────────────────────────────────────────────────────────────
    story.append(H1('Results', 6))
    story.append(section_line())

    story.append(H2('6.1  Training Loss'))
    story += fig(f'{FIG}/fig_pinn_loss.png', width=14.5*cm,
                 caption_text='Figure 2. PINN training loss over 10 000 epochs. '
                              'Left: full run on log scale. Right: second half, showing stable convergence.')

    story.append(H2('6.2  Quantitative Evaluation'))
    story.append(P(
        '<b>Training vs. evaluation split.</b> '
        'PINNs do not use a conventional supervised train/test split. '
        'During training, the model never sees labeled (S, τ, V) pairs for interior points — '
        'it only minimises the PDE residual, which requires no ground-truth labels. '
        'Supervised labels are used only at the terminal condition (τ = 0) and boundary points. '
        'After training, the model is evaluated on a separate, independent '
        '<b>100×100 = 10 000-point regular grid</b> that was not used in any form during training. '
        'Ground-truth prices at these grid points are computed with the analytical Black-Scholes formula. '
        'The rMAE and rRMSE values below therefore represent genuine out-of-sample generalisation error.'))
    story.append(P(
        'Evaluation metrics are computed on the 100×100 grid excluding τ < 10⁻⁴:'))
    story.append(Math('rMAE = Σ|V_pred − V_true| / Σ|V_true|'))
    story.append(Math('rRMSE = √(mean(V_pred − V_true)²) / mean|V_true|'))
    story.append(Spacer(1, 0.3*cm))

    story.append(KeepTogether([
        result_box([
            ['Model',          'Baseline PINN (European Call)'],
            ['rMAE',           '0.45%'],
            ['rRMSE',          '0.35%'],
            ['Training time',  '151 s'],
            ['Epochs',         '10 000'],
            ['Seed',           '42'],
        ]),
        Caption('Table 6. Baseline PINN quantitative results.')
    ]))

    story.append(H2('6.3  Price Surface Comparison'))
    story += fig(f'{FIG}/fig_pinn_vs_analytical.png', width=15.5*cm,
                 caption_text='Figure 3. Left: analytical Black-Scholes surface. Centre: PINN prediction. '
                              'Right: absolute error |V_pred − V_true|. The PINN accurately reproduces '
                              'the general shape; largest errors appear near the strike and at short maturities.')

    story.append(H2('6.4  Error Heatmap'))
    story += fig(f'{FIG}/fig_pinn_heatmap.png', width=10*cm,
                 caption_text='Figure 4. Absolute error heatmap on the (S, τ) grid. The dashed line marks '
                              'K = 100. Errors are concentrated near S ≈ K and at small τ, confirming '
                              'the spectral-bias problem.')

    story.append(H2('6.5  Slice Comparison at Fixed Maturities'))
    story += fig(f'{FIG}/fig_pinn_slices.png', width=15.5*cm,
                 caption_text='Figure 5. Analytical vs. PINN option price at τ ∈ {0.1, 0.3, 0.5, 1.0} years. '
                              'The PINN tracks the analytical solution closely; deviations are largest '
                              'near at-the-money (S ≈ K = 100) at the shortest maturity τ = 0.1.')

    # ── 7. Summary ────────────────────────────────────────────────────────────
    story.append(H1('Summary & Next Steps', 7))
    story.append(section_line())

    story.append(H2('7.1  Week 1–2 Accomplishments'))
    acc_headers = ['Task', 'Status']
    acc_rows = [
        ['PINNsFormer repository cloned, original models verified',        '✓'],
        ['Domain transfer (x,t) → (S,τ) implemented',                     '✓'],
        ['Synthetic call & put datasets generated via analytical B-S',     '✓'],
        ['Put-call parity verified  (max error ≈ 10⁻¹⁴)',                 '✓'],
        ['Baseline PINN trained and evaluated (rMAE, rRMSE)',              '✓'],
        ['Price surface, heatmap, and slice comparisons produced',         '✓'],
    ]
    story.append(simple_table(acc_headers, acc_rows, col_widths=[13*cm, 2.5*cm]))
    story.append(Caption('Table 7. Week 1–2 task completion summary.'))

    story.append(H2('7.2  Key Observations'))
    story.append(B(
        'The standard PINN reproduces the call price surface with <b>rMAE = 0.45%</b> in 151 s, '
        'confirming it is a strong and fast baseline.'))
    story.append(B(
        'Largest errors are concentrated <b>near the strike price</b> (S ≈ K = 100) and at '
        '<b>short maturities</b> (τ → 0) — exactly where the payoff has a kink.'))
    story.append(B(
        'This confirms the <b>spectral-bias</b> problem and motivates PINNsFormer\'s WaveAct '
        'activation and sequential loss.'))

    story.append(H2('7.3  Week 3–4 Plan'))
    story.append(B('Train BS_PINNsFormer (Finite-Difference and Autograd variants) and '
                   'compare against the PINN baseline.'))
    story.append(B('Hyperparameter sweep: λ₁, λ₂, λ₃; d_model; number of Transformer layers.'))
    story.append(B('Comprehensive evaluation with rMAE and rRMSE metrics.'))
    story.append(B('Wall-clock training time benchmark: PINN vs. PINNsFormer-FD vs. PINNsFormer-AD.'))

    # ── References ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(H1('References', 8))
    story.append(section_line())
    refs = [
        '[1] M. Raissi, P. Perdikaris, G. E. Karniadakis. "Physics-informed neural networks: '
        'A deep learning framework for solving forward and inverse problems involving nonlinear '
        'partial differential equations." <i>Journal of Computational Physics</i>, 378, 686–707, 2019.',
        '[2] W. Zhao, L. Zhang, Y. Xu. "PINNsFormer: A Transformer-Based Framework For '
        'Physics-Informed Neural Networks." <i>ICLR 2024</i>.',
        '[3] M. S. Dhiman, J. Hu. "Solving the Black-Scholes equation with Physics-Informed '
        'Neural Networks." arXiv, 2023.',
        '[4] S. M. Nuugulu et al. "Physics-informed versus data-driven neural networks for '
        'option pricing under the Black-Scholes framework." '
        '<i>Applied Mathematics and Computation</i>, 2025.',
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('Ref', fontSize=9, fontName='Helvetica',
                                                  leading=13, textColor=GRAY,
                                                  spaceAfter=5, leftIndent=12, firstLineIndent=-12)))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f'PDF saved → {OUT}')

if __name__ == '__main__':
    build()
