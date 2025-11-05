# Sergei F. Vyboishchikov, Universitat de Girona, October 2025
# Calculation of atomic charges from descriptors by Kocer-Mason-Erturk, doi: 10.1063/1.5086167
# CatBoost machine-learning is used
# Usage: 
#      BoostCharge-generator.py -Files file1.xyz[,file2.xyz]
# where file1.xyz[,file2.xyz] is a comma-separated list of XYZ input files
# or
#      BoostCharge-generator.py -Directory DIR
# where DIR is a directory of XYZ input files. The ".xyz" file extension is obligatory
# Output: .BoostCha-charges files (one file for each input XYZ file). Located in the same directory as XYZ files.
import os, sys, scipy.special, numpy as np, catboost

rc = 3.0 # rc in the cutoff radius
nmax = 4
lmax = 5
Depth = 2

FilesToRead = ''
DirectoryToRead = ''

for i in range(1,len(sys.argv)-1):
   if sys.argv[i].lower()=='-directory':
      DirectoryToRead = sys.argv[i+1]
   elif sys.argv[i].lower() in ['-files', '-file']:
      FilesToRead = sys.argv[i+1]

Elements  = '0-H-He-Li-Be-B-C-N-O-F-Ne-Na-Mg-Al-Si-P-S-Cl-Ar-K-Ca-Sc-Ti-V-Cr-'
Elements += 'Mn-Fe-Co-Ni-Cu-Zn-Ga-Ge-As-Se-Br-Kr-Rb-Sr-Y-Zr-Nb-Mo-Tc-Ru-Rh-Pd-'
Elements += 'Ag-Cd-In-Sn-Sb-Te-I-Xe-Cs-Ba-La-Ce-Pr-Nd-Pm-Sm-Eu-Gd-Tb-Dy-Ho-Er-'
Elements += 'Tm-Yb-Lu-Hf-Ta-W-Re-Os-Ir-Pt-Au-Hg-Tl-Pb-Bi-Po-At-Rn-Fr-Ra-Ac-Th-'
Elements += 'Pa-U-Np-Pu-Am-Cm-Bk-Cf-Es-Fm-Md-No-Lr-Rf-Db-Sg-Bh-Hs-Mt-Ds-Rg-Cn-'
Elements += 'Nh-Fl-Mc-Lv-Ts-Og'
Elem = {} # Dictionary of element numbers. For example Elem['C'] = 6 and Elem['6'] = 6, so that both element names and number are understood
for i,r in enumerate(Elements.split('-')):
   Elem[r] = i
   Elem[str(i)] = i
for r in 'Si','Sn','Pb','14','50','82':
   Elem[r] = 6

def ReadXYZCoordinates(directory,file_name,data):
   Qtot = 0
   XYZ = []
   el = []
   for i,row in enumerate(data):
      if len(row.split())>=4:
         el.append(Elem[row.split()[0]])
         XYZ.append(row.split()[1:4])
      elif i!=1 and len(row.split())==2:
         Qtot = int(row.split()[0])
   el = np.array(el,dtype="int")
   XYZ = np.array(XYZ,dtype="float32")
   return {"Directory":directory,"FileName":file_name,"Natoms":len(el), "XYZ": XYZ, "el":el, "Qtot":Qtot}

def sinc(x): 
   if x==0:
      return 1.0
   else:
      return np.sin(x)/x

e = np.zeros(nmax+1)
d = np.ones(nmax+1)
for n in range(1,nmax+1):
   e[n] = (n*(n+2))**2/(4*(n+1)**4+1)
   d[n] = 1-e[n]/d[n-1]

def Radial_gn(rc,nmax,r):
   g = np.zeros(nmax+1)
   if r>rc:
      return g
   pir_rc = np.pi*r/rc
   f = np.zeros(nmax+1)
   prefix = np.pi*np.sqrt(2.0/rc)/rc
   if r==0:
      f[0] = prefix*4/np.sqrt(5.0)
   else:
      f[0] = np.sqrt(2/(5*rc))*(2*np.sin(pir_rc)+np.sin(2*pir_rc))/r
   g[0] = f[0]
   for n in range(1,nmax+1):
      prefix = -prefix
      f[n] = prefix*(n+1)*(n+2)/np.sqrt((n+1)**2+(n+2)**2) * (sinc(pir_rc*(n+1)) + sinc(pir_rc*(n+2)))
      g[n] = (f[n]+np.sqrt(e[n]/d[n-1])*g[n-1])/np.sqrt(d[n])
   return g

def CartesianToSpherical(XYZ):
   r = np.sqrt(np.sum(XYZ**2))
   theta = np.arccos(XYZ[2]/r) if r != 0 else 0  # handle case when r = 0
   phi = np.arctan2(XYZ[1],XYZ[0])
   return r, theta, phi

def CalculatePNL(Atoms,XYZ,nmax,lmax):
   NAtoms = XYZ.shape[0]
   PNL = np.zeros((NAtoms,nmax+1,lmax+1))
   for i in range(NAtoms):
      Coef = np.zeros((nmax+1,lmax+1,2*lmax+1),dtype=np.cdouble)
      for j in list(range(i))+list(range(i+1,NAtoms)): # i!=j
         rij, thetaij, phiij = CartesianToSpherical(XYZ[j]-XYZ[i])
         gn = Atoms[j]*Radial_gn(rc,nmax,rij) # contains g_n[n=0...nmax]
         YLM = scipy.special.sph_harm_y_all(lmax, lmax, thetaij, phiij)
         Coef += gn[:,None,None] * YLM[None,:,:]
      PNL[i] = np.sum(np.abs(Coef)**2,axis=-1)
   return PNL.reshape(NAtoms,-1)

def ReadMoleculeFromDirectory(directory,dire=True):
   mol_list = []
   if dire:
      file_list = os.listdir(directory)
   else:
      file_list = directory
      directory = ''
   for file_name in file_list:
      if file_name.split('.')[-1]=='xyz':
         file_path = os.path.join(directory, file_name)
         if os.path.isfile(file_path):
            try:
               with open(file_path, 'r') as File:
                  mol = ReadXYZCoordinates(directory,file_name,File.read().split('\n'))
                  mol_list.append(mol)
            except:
               print("File "+file_path+" cannot be read")
   return mol_list

def CalculateChargeDescriptors(X,mol_list,Q_prime):
   DescriptorNames = [
   'ACI', # average charge ideal = mol["Qtot"]/mol["Natoms"]
   'AC', # average charge
   'MPC',# mean of positive charges
   'MNC',# mean of negative charges
   'MAC',# mean absolute charge
  'RPC',# relative positive charge
  'RNC',# relative negative charge
  ]
   atoms_counter = 0
   Descriptors = np.zeros((len(Q_prime),1+1+len(DescriptorNames))) # +5
   for mol in mol_list:
      X_mol = np.array(X[atoms_counter:atoms_counter+mol["Natoms"]])
      Q_prime_mol = np.array([Q_prime[atoms_counter:atoms_counter+mol["Natoms"]]]).T
      ACI = mol["Qtot"]/mol["Natoms"]
      AC = sum(Q_prime_mol)/mol["Natoms"]
      MPC = sum(Q_prime_mol[Q_prime_mol>0])
      MNC = sum(Q_prime_mol[Q_prime_mol<0])
      MAC = sum(abs(Q_prime_mol))
      RPC = 0
      RNC = 0
      if(MPC!=0): RPC = np.max(Q_prime_mol)/MPC
      if(MNC!=0): RNC = np.max(Q_prime_mol)/MNC
      MPC /= mol["Natoms"]
      MNC /= mol["Natoms"]
      MAC /= mol["Natoms"]
      result = np.hstack((
        X_mol[:,0:1],
        Q_prime_mol,
        np.tile(np.array([AC[0],ACI,MPC,MNC,MAC[0],RPC,RNC])[0:len(DescriptorNames)], (mol["Natoms"], 1)),
      ))
      Descriptors[atoms_counter:atoms_counter+mol["Natoms"], :] = result
      atoms_counter += mol["Natoms"]
   return Descriptors

def MakeVector(mol_list):
   Natom = sum([mol["Natoms"] for mol in mol_list])
   atoms_counter = 0
   X = np.zeros((Natom,(nmax+1)*(lmax+1)+2))
   for mol in mol_list:
      pnl = CalculatePNL(mol["el"], mol["XYZ"], nmax,lmax)
      result = np.hstack((mol["el"][:,None], np.full((mol["Natoms"],1),mol["Qtot"]/mol["Natoms"]), pnl))
      X[atoms_counter:atoms_counter+mol["Natoms"]] = result 
      atoms_counter += mol["Natoms"]
   return X

def CalculateCorrectedCharges(mol_list,Q_prime):
   atoms_counter = 0
   Q = np.zeros_like(Q_prime)
   for mol in mol_list:
      Qtot_prime = sum(Q_prime[atoms_counter:atoms_counter+mol["Natoms"]])
      Q[atoms_counter:atoms_counter+mol["Natoms"]] = Q_prime[atoms_counter:atoms_counter+mol["Natoms"]] - (Qtot_prime-mol["Qtot"])/mol["Natoms"]
      atoms_counter += mol["Natoms"]
   return Q

def WriteChargeFiles(mol_list,Y):
   atoms_counter = 0
   for mol in mol_list:
      ChargeFileName = '.'.join(mol["FileName"].split('.')[:-1])
      with open(os.path.join(mol["Directory"], ChargeFileName+'.BoostCha-charges'),'w') as File:
         File.write(' QDAT for System : '+ChargeFileName.strip()+'\n Atomic coordinates\n')
         for i in range(mol["Natoms"]):
            File.write("%4i  "%mol["el"][i]+(3*"%12.4f")%(tuple(mol["XYZ"][i]))+'\n')
         File.write('\n Ground state charges\n')
         for i in range(mol["Natoms"]):
            File.write("%8.4f"%Y[atoms_counter+i])
            if (i+1)%10==0:
               File.write('\n')
         File.write('\n')
      atoms_counter += mol["Natoms"]

if FilesToRead == '' and DirectoryToRead != '':
   mol_list = ReadMoleculeFromDirectory(DirectoryToRead)
elif FilesToRead != '' and DirectoryToRead == '':
   mol_list = ReadMoleculeFromDirectory(FilesToRead.split(','),dire=False)
else:
   print('Either "-directory" or "-files" option must be provided')
   quit()

X = MakeVector(mol_list)

model = catboost.CatBoostRegressor()
model.load_model('BoostCharge-generator-model-1.json',format='json')
model2 = catboost.CatBoostRegressor()
model2.load_model('BoostCharge-generator-model-2.json',format='json')

predictions = model.predict(X)
print("\nFirst step done.",end='')
Descriptors = CalculateChargeDescriptors(X,mol_list,predictions)
predictions = model2.predict(Descriptors)
print("   Second step done.")

predictions_corrected = CalculateCorrectedCharges(mol_list,predictions) # corrected charges
print(len(X),"charges in",len(mol_list),"molecules have been calculated.")
WriteChargeFiles(mol_list,predictions_corrected)
print("Finished!")