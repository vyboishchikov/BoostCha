# BoostCha – Rapid prediction of atomic charges based on gradient boosting  
*Sergei F. Vyboishchikov, Universitat de Girona, October 2025*  

The **BoostCha** model operates in two steps: it first predicts pseudo-charges for individual atoms based on their local environments, represented by three-dimensional descriptors of Kocer–Mason–Erturk type (doi: [10.1063/1.5086167](https://pubs.aip.org/aip/jcp/article-abstract/150/15/154102/76113/A-novel-approach-to-describe-chemical-environments?redirectedFrom=fulltext)), and then refines these values using global molecular information. The model uses the **CatBoost** machine-learning module.

---

**Usage**:
`BoostCharge-generator.py -Files` *file1.xyz[,file2.xyz]*

where *file1.xyz[,file2.xyz]* is a comma-separated list of XYZ input files  

   or
```
BoostCharge-generator.py -Directory *DIR*
```
where *DIR* is a directory of XYZ input files. The `.xyz` file extension is obligatory.

**Output:** `.BoostCha-charges` files (one for each input XYZ file). They will be located in the same directory as the XYZ files.  

**Python modules required:** `os`, `sys`, `Scipy`, `NumPy`, `CatBoost`.

**Download** the [BoostCharge-generator.py](https://github.com/vyboishchikov/BoostCha/blob/main/BoostCharge-generator.py) Python code.

The paper is submitted.
