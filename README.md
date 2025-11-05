# BoostCha &ndash; Rapid prediction of atomic charges based on gradient boosting
 <i>Sergei F. Vyboishchikov, Universitat de Girona, October 2025</i>

<p>The BoostCha model operates in two steps: it first predicts pseudo-charges for individual atoms based on their local environments, represented by three-dimensional descriptors of Kocer–Mason–Erturk type (doi: 10.1063/1.5086167), and then refines these values using global molecular information. The model uses CatBoost machine-learning module</p>
<p><p><b>Usage</b>: 
<br><code>      BoostCharge-generator.py -Files <i>file1.xyz[,file2.xyz]</i></code>
<br> where <i>file1.xyz[,file2.xyz]</i> is a comma-separated list of XYZ input files
<br>    or
<br><code>      BoostCharge-generator.py -Directory <i>DIR</i></code>
<br> where <i>DIR</i> is a directory of XYZ input files. The <code>.xyz</code> file extension is obligatory.
<p> <b>Output</b>: <code>.BoostCha-charges</code> files (one file for each input XYZ file). They will be located in the same directory as the XYZ files.
<p> Python modules required: <code>os</code>, <code>sys</code>, <code>Scipy</code>, <code>NumPy</code>, <code>CatBoost</code>. </p>
<p><p>Download BoostCharge-generator.py Python code <a href="https://github.com/vyboishchikov/BoostCha/blob/main/BoostCharge-generator.py">here</a>.

<p><p>The paper is submitted.

