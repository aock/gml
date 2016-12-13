from __future__ import division
import pickle
import sys
import numpy as np

if __name__ == "__main__":

	data = None

	p_out_deg1 = []
	p_out_deg2 = []
	p_out_deg3 = []
	p_out_deg4 = []

	p_in_deg1 = []
	p_in_deg2 = []
	p_in_deg3 = []
	p_in_deg4 = []

	g_out_deg2 = []
	g_out_deg3 = []
	g_out_deg4 = []

	g_in_deg2 = []
	g_in_deg3 = []
	g_in_deg4 = []

	for i in range(1,11,1):
		filename = "data" + str(i)

		with open(filename, "rb") as f:
			data = pickle.load(f)
			for el in data:
				p_out_deg1.append(el[0])
				p_out_deg2.append(el[1])
				p_out_deg3.append(el[2])
				p_out_deg4.append(el[3])
								  
			data = pickle.load(f)
			for el in data:
				p_in_deg1.append(el[0])
				p_in_deg2.append(el[1])
				p_in_deg3.append(el[2])
				p_in_deg4.append(el[3])
								 
			data = pickle.load(f)
			for el in data:
				g_out_deg2.append(el[0])
				g_out_deg3.append(el[1])
				g_out_deg4.append(el[2])
								 
			data = pickle.load(f)
			for el in data:
				g_in_deg2.append(el[0])
				g_in_deg3.append(el[1])
				g_in_deg4.append(el[2])
		print("Processed:", filename)

				
	print("1d - Poly-Eout:", np.sum(p_out_deg1) / len(p_out_deg1))
	print("2d - Poly-Eout:", np.sum(p_out_deg2) / len(p_out_deg2))
	print("3d - Poly-Eout:", np.sum(p_out_deg3) / len(p_out_deg3))
	print("4d - Poly-Eout:", np.sum(p_out_deg4) / len(p_out_deg4))

	print("1d - Poly-Ein:", np.sum(p_in_deg1) / len(p_in_deg1))
	print("2d - Poly-Ein:", np.sum(p_in_deg2) / len(p_in_deg2))
	print("3d - Poly-Ein:", np.sum(p_in_deg3) / len(p_in_deg3))
	print("4d - Poly-Ein:", np.sum(p_in_deg4) / len(p_in_deg4))

	print("2p - GLT-Eout:", np.sum(g_out_deg2) / len(g_out_deg2))
	print("3p - GLT-Eout:", np.sum(g_out_deg3) / len(g_out_deg3))
	print("4p - GLT-Eout:", np.sum(g_out_deg4) / len(g_out_deg4))

	print("2p - GLT-Ein:", np.sum(g_in_deg2) / len(g_in_deg2))
	print("3p - GLT-Ein:", np.sum(g_in_deg3) / len(g_in_deg3))
	print("4p - GLT-Ein:", np.sum(g_in_deg4) / len(g_in_deg4))

	print("Gesamt Data: ", len(p_out_deg1))
        


