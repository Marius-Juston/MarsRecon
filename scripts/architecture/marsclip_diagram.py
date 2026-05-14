import graphviz


def generate_marsclip_journal_quality():
    # Initialize directed graph with academic typesetting standards
    dot = graphviz.Digraph(name="MarsCLIP_Architecture_Journal", format="pdf")
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.8',
             fontname='Helvetica', compound='true', dpi='300', bgcolor='transparent')

    # Base node & edge styling: Minimalist, muted palettes, professional borders
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#FFFFFF', color='#343A40',
             fontname='Helvetica', fontsize='11', penwidth='1.2', margin='0.15,0.1')
    dot.attr('edge', fontname='Helvetica', fontsize='9', color='#495057', penwidth='1.0', arrowsize='0.7')

    # ==========================================
    # PHASE 0: Data Infrastructure
    # ==========================================
    with dot.subgraph(name='cluster_phase0') as c0:
        c0.attr(label='Phase 0: Data Infrastructure', style='solid', color='#DEE2E6', bgcolor='#F8F9FA',
                fontname='Helvetica-Bold')

        c0.node('PDS', 'NASA PDS HiRISE RDR\n(Raw DNs, Metadata)', shape='cylinder', fillcolor='#E3F2FD')
        c0.node('COG', 'COG Conversion &\nGeographic CRS Reproj', fillcolor='#E3F2FD')
        c0.node('Sampler', 'Strip-Aware Spatial Sampler\n(Single Crop I\u1d62)', fillcolor='#E3F2FD')
        c0.node('Calib', 'Radiometric Calibration\n(I/F Reflectance, Nodata Mask)', fillcolor='#E3F2FD')

        c0.edge('PDS', 'COG')
        c0.edge('COG', 'Sampler', xlabel=' Footprints ')
        c0.edge('Sampler', 'Calib', xlabel=' Patches ')

    # ==========================================
    # STAGE A: Masked Autoencoder (MAE)
    # ==========================================
    with dot.subgraph(name='cluster_stageA') as ca:
        ca.attr(label='Stage A: Masked Autoencoder (MAE)', style='solid', color='#DEE2E6', bgcolor='#F4FCE3',
                fontname='Helvetica-Bold')

        ca.node('InputA', 'Input Patch\n(224×224×3)')
        ca.node('TokenA', 'Spectral Band Tokenization\n+ Nodata Dropping')
        ca.node('PosEncA', 'Scale-Aware Positional Encoding\nS(\u03C1) using MAP_SCALE')
        ca.node('MaskA', 'Random Masking (75%)')

        ca.node('ViTEncA', '<<b>ViT Encoder</b><br/>(Processes Visible Tokens Only)>',
                fillcolor='#D8F5A2', penwidth='1.5')
        ca.node('MaskTokens', 'Insert Learnable Mask Tokens\n+ Re-add Positional Encoding')
        ca.node('ViTDecA', '<<b>Lightweight ViT Decoder</b><br/>(Processes All Tokens)>',
                fillcolor='#D8F5A2')

        ca.node('LossA', 'MSE Reconstruction Loss\n(Masked, Valid Pixels Only)', shape='hexagon', fillcolor='#FFE3E3')

        ca.edge('InputA', 'TokenA')
        ca.edge('TokenA', 'PosEncA')
        ca.edge('PosEncA', 'MaskA')
        ca.edge('MaskA', 'ViTEncA', xlabel=' 25% Visible Tokens ')
        ca.edge('ViTEncA', 'MaskTokens', xlabel=' Latents (h\u1d62) ')
        ca.edge('MaskTokens', 'ViTDecA')
        ca.edge('ViTDecA', 'LossA', xlabel=' x\u0302\u1d62 ')
        ca.edge('InputA', 'LossA', style='dotted', xlabel=' x\u1d62 (Target) ', constraint='false')

    # ==========================================
    # STAGE B: Contrastive Multi-Modal Alignment
    # ==========================================
    with dot.subgraph(name='cluster_stageB') as cb:
        cb.attr(label='Stage B: Contrastive Multi-Modal Alignment', style='solid', color='#DEE2E6', bgcolor='#FFF4E6',
                fontname='Helvetica-Bold')

        # Encoders Subgraph
        with cb.subgraph(name='cluster_encoders') as ce:
            ce.attr(label='Independent Modality Encoders', style='dotted', color='#ADB5BD')

            with ce.subgraph() as level1:
                level1.attr(rank='same')
                level1.node('Crop', 'Image Crop\n(Single Scale I\u1d62)', shape='parallelogram', fillcolor='#F1F3F5')
                level1.node('GeoParams',
                            'Viewing Parameters\n(\u03B8\u1d62, \u03B8\u2091, \u03B8\u209A, t, L\u209B, \u03C6\u209B\u209B)',
                            shape='parallelogram', fillcolor='#F1F3F5')
                level1.node('LocCoords', 'Planetary Location\n(Lat, Lon)', shape='parallelogram', fillcolor='#F1F3F5')
                level1.node('TextDesc', 'Scientific Rationale\n(LLM Expanded)', shape='parallelogram',
                            fillcolor='#F1F3F5')

                cb.edge('Crop', 'GeoParams', style='invis')
                cb.edge('GeoParams', 'LocCoords', style='invis')
                cb.edge('LocCoords', 'TextDesc', style='invis')

            with ce.subgraph() as level2:
                level2.attr(rank='same')
                level2.node('EncV',
                            '<<b>Vision Encoder (E<sub>V</sub>)</b><br/>Pretrained MAE Encoder>',
                            fillcolor='#FFE8CC', penwidth='1.5')
                level2.node('EncG',
                            '<Geometry Encoder (E<sub>G</sub>)<br/>RFF + Periodic \u2192 MLP>',
                            fillcolor='#EBEBFF')

                level2.node('EncL',
                            '<Location Encoder (E<sub>L</sub>)<br/>Spherical Harmonics \u2192 SirenNet>',
                            fillcolor='#EBEBFF')

                level2.node('EncT',
                            '<Text Encoder (E<sub>T</sub>)<br/>Frozen Sentence Transformer>',
                            fillcolor='#EBEBFF')

            with ce.subgraph() as level3:
                level3.attr(rank='same')
                level3.node('FiLM', 'FiLM Layer\nModulation via g\u1d62', fillcolor='#EBEBFF')
                level3.node('g_i', 'g\u1d62 \u2208 \u211D\u1d48', shape='ellipse')
                level3.node('l_i', 'l\u1d62 \u2208 \u211D\u1d48', shape='ellipse')
                level3.node('t_i', 't\u1d62 \u2208 \u211D\u1d48', shape='ellipse')

        # Feature Fusion & Loss
        cb.node('v_vis', 'v\u1d62 \u2208 \u211D\u1d48', shape='ellipse')
        cb.node('CrossAttn', 'Cross-Attention Fusion\nz\u1d62 = LayerNorm(t\u1d62 + gate \u00B7 (\u03B1 V\u2097))',
                fillcolor='#FFD8A8')
        cb.node('z_i', 'Fused Context Target (z\u1d62)', shape='ellipse', fillcolor='#FFD8A8', penwidth='1.5')
        cb.node('LossSoft', 'Soft-Target InfoNCE\nKL( P(i,j) || softmax(v\u1d62 \u00B7 z\u2c7C / \u03C4) )',
                shape='hexagon', fillcolor='#FFE3E3')

        # Structural Routing
        cb.edge('Crop', 'EncV')
        cb.edge('GeoParams', 'EncG')
        cb.edge('LocCoords', 'EncL')
        cb.edge('TextDesc', 'EncT')

        cb.edge('EncV', 'FiLM')
        cb.edge('EncG', 'g_i')
        cb.edge('EncL', 'l_i')
        cb.edge('EncT', 't_i')

        cb.edge('g_i', 'FiLM', xlabel=' \u0394\u03B3, \u0394\u03B2 ', style='dashed')
        cb.edge('FiLM', 'v_vis')

        cb.edge('l_i', 'CrossAttn', xlabel=' Keys / Values ')
        cb.edge('t_i', 'CrossAttn', xlabel=' Queries / Residual ')
        cb.edge('CrossAttn', 'z_i')

        cb.edge('v_vis', 'LossSoft', xlabel=' v\u1d62 ')
        cb.edge('z_i', 'LossSoft', xlabel=' z\u2c7C ')

    # ==========================================
    # Global System Connections
    # ==========================================
    dot.edge('Calib', 'InputA', style='bold', color='#495057')
    dot.edge('Calib', 'Crop', style='bold', color='#495057')
    dot.edge('ViTEncA', 'EncV', style='bold', color='#2F9E44', xlabel=' Transfer Pretrained Weights ',
             constraint='false')

    dot.render('MarsCLIP', cleanup=True)
    print("Journal-quality diagram generated: MarsCLIP.pdf")


if __name__ == '__main__':
    generate_marsclip_journal_quality()
