<mxfile host="app.diagrams.net">
  <diagram name="Ensemble Deployment" id="QWERTY1234567">
    <mxGraphModel dx="827" dy="494" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <mxCell id="2" value="User Input (Image + Method)" style="shape=ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1">
          <mxGeometry x="300" y="20" width="180" height="60" as="geometry" />
        </mxCell>

        <mxCell id="3" value="Ensemble Aggregator" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
          <mxGeometry x="300" y="120" width="180" height="60" as="geometry" />
        </mxCell>

        <mxCell id="4" value="Bagging (Simple Average)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1">
          <mxGeometry x="80" y="250" width="160" height="60" as="geometry" />
        </mxCell>

        <mxCell id="5" value="Boosting (Weighted Average)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1">
          <mxGeometry x="300" y="250" width="160" height="60" as="geometry" />
        </mxCell>

        <mxCell id="6" value="Stacking (Meta Model)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1">
          <mxGeometry x="520" y="250" width="160" height="60" as="geometry" />
        </mxCell>

        <mxCell id="7" value="A/B Testing (Bagging vs Boosting)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;" vertex="1" parent="1">
          <mxGeometry x="190" y="370" width="180" height="60" as="geometry" />
        </mxCell>

        <mxCell id="8" value="Fallback Mechanism (If Stacking fails)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;" vertex="1" parent="1">
          <mxGeometry x="410" y="370" width="200" height="60" as="geometry" />
        </mxCell>

        <mxCell id="9" value="Final Prediction Output" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;" vertex="1" parent="1">
          <mxGeometry x="300" y="480" width="180" height="60" as="geometry" />
        </mxCell>

        <!-- Connecting Arrows -->
        <mxCell id="10" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;" edge="1" parent="1" source="2" target="3">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <mxCell id="11" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;" edge="1" parent="1" source="3" target="4">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="180" y="210" as="targetPoint" />
          </mxGeometry>
        </mxCell>

        <mxCell id="12" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;" edge="1" parent="1" source="3" target="5">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <mxCell id="13" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;" edge="1" parent="1" source="3" target="6">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="600" y="210" as="targetPoint" />
          </mxGeometry>
        </mxCell>

        <!-- Only Bagging and Boosting go to A/B Testing -->
        <mxCell id="14" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;dashed=1;" edge="1" parent="1" source="4" target="7">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <mxCell id="15" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;dashed=1;" edge="1" parent="1" source="5" target="7">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- Only Stacking goes to Fallback -->
        <mxCell id="16" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;dashed=1;" edge="1" parent="1" source="6" target="8">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- From A/B Testing and Fallback to Final Output -->
        <mxCell id="17" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;" edge="1" parent="1" source="7" target="9">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <mxCell id="18" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;" edge="1" parent="1" source="8" target="9">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
