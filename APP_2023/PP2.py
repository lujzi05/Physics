import re
import os
import numpy as np
import time
from matplotlib import pyplot as plt



def extracting_tags(text_str, tag):
	"""
	Extract any tag from a string to a list.
	"""
	reg_str = "<" + tag + ">([\S\s]*?)</" + tag + ">" #(.*?)[\S+\n\r\s]
	res = re.findall(reg_str, text_str)
	return res


def alphaS_Q(text):
	"""
	Get the values of Q and alphaS for all events.
	"""
	alphaS = [float(event.split()[5]) for event in extracting_tags(text, "event")]
	Q = [float(rscale.split()[-1]) for rscale in extracting_tags(text, "rscale")]
	return Q, alphaS


def analyze_event(text, inv_part=False):
	"""
	Extract the missing transverse energy from every event.
	If inv_part is True: write out the occuring invisible particles in the final state.
	"""
	events = extracting_tags(text, "event")
	missing_E_ts = []
	if os.path.exists("invisible_particles.txt") and inv_part:
		os.remove("invisible_particles.txt")
	for i in range(len(events)):
		number_of_particles = int(events[i].splitlines()[1].split()[0])
		event_list = [line.split() for line in events[i].splitlines()[2:]][:number_of_particles]
		event_array = np.array([np.float_(ii) for ii in event_list])
		
		invisible_particles_dict = {'DM':6000022, 'v_e':12, 'v_m':14, 'v_t':16, 'anti_v_e':-12, 'anti_v_m':-14, 'anti_v_t':-16}
		final_particles_array = event_array[event_array[:,1]==1][:,[0,6,7]]
		inv_part_mask = np.isin(final_particles_array[:,0], list(invisible_particles_dict.values()))
		
		# calculate the missing transverse energy
		missing_E_ts.append(np.sqrt(np.square(final_particles_array[~inv_part_mask,1:].sum(axis=0)).sum()))
		
		# it writes out the invisible particles of every event to a file (it doubles the runtime)
		if inv_part:
			inv_part_pdg = final_particles_array[inv_part_mask,0]
			inv_part_name = [k for k, v in invisible_particles_dict.items() if v in inv_part_pdg]
			with open("invisible_particles.txt", "a") as f:
				f.write("Event " + str(i+1) + ": " + str(inv_part_name)[1:-1] + "\n")
		
	return missing_E_ts

	
def cut_energy(energy, treshold):
	"""
	Remove every item from the energy list which is lower than the treshold.
	"""
	energy_array = np.array(energy)
	return energy_array[energy_array>treshold]



def significance(init_sigma, threshold_list, missing_E_ts, missing_E_ts_bg, L=300000):
	"""
	Calculate the significance for a list of thresholds given an initial cross-section.
	"""
	sigma_signal = np.array([len(cut_energy(missing_E_ts,threshold))/len(missing_E_ts)*init_sigma[0] for threshold in threshold_list])
	sigma_background = np.array([len(cut_energy(missing_E_ts_bg,threshold))/len(missing_E_ts_bg)*init_sigma[1] for threshold in threshold_list])

	z = np.sqrt(L)*sigma_signal/(np.sqrt(sigma_signal+sigma_background))
	return z




def main():
	start = time.time()

	# open data
	with open("./Signal/Events/BP1/unweighted_events.lhe") as f:
		text = f.read()

	with open("./Signal/Events/BP2/unweighted_events.lhe") as f:
		text2 = f.read()

	with open("./Background/Events/BP1_background/unweighted_events.lhe") as f:
		bg = f.read()

	# plot alpha_S in terms of Q
	Q, alphaS = alphaS_Q(text)
	plt.plot(Q, alphaS, '.', label=r'$\alpha_S(Q)$', markersize=2, color='darkred')
	plt.xlabel('Q')
	plt.ylabel(r'$\alpha_S$')
	plt.legend()
	plt.savefig('alphaS_Q.png')
	plt.close()


	# plot histogram from 0 GeV to 1000 GeV with binning 25 GeV (bins=1000/25=40)
	missing_E_ts = analyze_event(text, inv_part=True)
	plt.hist(missing_E_ts, bins=40, alpha=0.5, density=True, range=(0,1000), color='darkred', edgecolor='black', linewidth=0.5)
	plt.xlim([0, 1000])
	plt.xlabel(r'$E_T^{miss} (GeV)$')
	plt.savefig('missing_E_t.png')
	plt.close()


	# plot histogram from 0 GeV to 1000 GeV with binning 25 GeV (bins=1000/25=40) with background
	missing_E_ts_bg = analyze_event(bg)
	plt.hist(missing_E_ts, bins=40, alpha=0.5, density=True, range=(0,1000), color='darkred', edgecolor='black', linewidth=0.5, label='signal')
	plt.hist(missing_E_ts_bg, bins=40, alpha=0.3, density=True, range=(0,1000), color='cadetblue', edgecolor='black', linewidth=0.5, label='background')
	plt.xlim([0, 1000])
	plt.legend()
	plt.xlabel(r'$E_T^{miss} (GeV)$')
	plt.savefig('missing_E_t_bg.png')
	plt.close()







	# after cut treshold at 150GeV, the remaining number of events
	signal = len(cut_energy(missing_E_ts,150))
	background = len(cut_energy(missing_E_ts_bg,150))
	print('The remaining number of events after cutting the signal at 150GeV:'+str(signal)+'\nThe remaining number of events after cutting the background at 150GeV:'+str(background))




	threshold_list = np.linspace(0,1000, 1000)
	

	# significance for the BP1 simulation
	init_sigma = [1.110787*10**(-3), 4.113035 *10**2] # [signal, background]
	z = significance(init_sigma, threshold_list, missing_E_ts, missing_E_ts_bg)
	plt.plot(threshold_list, z, '.', markersize=3, color='black', label='BP1')
	plt.grid()
	plt.legend()
	plt.xlabel(r'$E_T^{miss} (GeV)$')
	plt.ylabel('significance z')
	plt.savefig('significance.png')
	plt.close()


	# significance for the BP2 simulation
	init_sigma = [2.449408*10**(-12), 4.113035 *10**2] # [signal, background]
	missing_E_ts2 = analyze_event(text2)
	z = significance(init_sigma, threshold_list, missing_E_ts, missing_E_ts_bg)
	plt.plot(threshold_list, z, '.', markersize=3, color='black', label='BP2')
	plt.grid()
	plt.legend()
	plt.xlabel(r'$E_T^{miss} (GeV)$')
	plt.ylabel('significance z')
	plt.savefig('significance2.png')
	plt.close()


	end = time.time()-start
	print('runtime:' + str(end))




if __name__ == "__main__":
    main()

