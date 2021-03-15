#CZAJA_sample
#Code to read HDF5 output created across mutliple cores and collect into a single domain for analysis
#This code is parallelized to read in mulitple Nprocs at the same time
#This code is NOT parallelized on time
#If the number of run cores ==1 then the program will read in serial automatically
import numpy as np
import h5py
from multiprocessing import Pool
from functools import partial
import timeit
import os

def read_hdf5_atomicblock(outputs,datapath,name,time,n):
	"""Per atomic block function to read specified data from a single HDF5 file'''

    Keyword arguments:
    outputs -- list of strings specifying the output to be read
	datapath -- string for path to hdf5 data i.e. /path/to/hdf5
	name -- string name of type of data i.e. Fluid, RBC, or PLT
	time -- integer for timestep of data to be read in
	n -- integer denoting core of HDF5 file
	"""

	#open HDF5 file
	#Read HemoCell v2.0 and v1.0 output. 
	try:
		#HemoCell V2.0 output format
		hdf5datafile = h5py.File(datapath+"/"+str(time).zfill(12)+"/"+name+"." + str(time).zfill(12) +".p." +str(n) +".h5","r")	
	except:
		try:
			#HemoCell V1.0 output format
			hdf5datafile = h5py.File(datapath+"/"+str(time)+"/"+name+"." + str(time) +".p." +str(n) +".h5","r")
		except (OSError, IOError):
			#If file does not exist raise the error
			raise
	
	#Append data per output string to a dictionary
	data = {}
	for output in outputs:
		#If data is LBM Fluid each output needs to be reshaped to be analyzed over the entire domain
		if "Fluid" in name:
			attribute = []
			tempattribute = np.array(hdf5datafile[output])

			#Reshape each output attribute so it can be indexed using numpy indexing
			if "Position" in data.keys():
				#X and Y indicies are reversed for better visualization with Paraview
				xblocks = np.shape(tempattribute)[2]
				yblocks = np.shape(tempattribute)[1]
				zblocks = np.shape(tempattribute)[0]
				for xpos in range(xblocks):
					for ypos in range(yblocks):
						for zpos in range(zblocks):
							attribute.append(tempattribute[zpos][ypos][xpos])
				data[output] = np.array(attribute)

			#Create a Position output to identfy the location of each LBM in the entire domain 
			else:
				position =[]
				#Get reletive postion of each atomic block
				relpos = hdf5datafile.attrs.get('relativePosition')
				#X and Y indicies are reversed for better visualization with Paraview
				xblocks = np.shape(tempattribute)[2]
				yblocks = np.shape(tempattribute)[1]
				zblocks = np.shape(tempattribute)[0]
				for xpos in range(xblocks):
					for ypos in range(yblocks):
						for zpos in range(zblocks):
							attribute.append(tempattribute[zpos][ypos][xpos])
							position.append(np.array([xpos+relpos[2],ypos+relpos[1],zpos+relpos[0]]))

				data[output] = np.array(attribute)
				data["Position"] = np.array(position)

		#If data is Cell type simply append it to a dictionary 
		else:
			data[output] = np.array(hdf5datafile[output])

	hdf5datafile.close()

	#Return desired data as a dictionary over atomic block domain
	return(data)

def open_hdf5_files(outputs=[""],datapath=".",name="",time=0,Nrunprocs=1,Nreadprocs=1):
	"""Function to read specified data from all N HDF5 files per timestep'''

    Keyword arguments:
    outputs -- list of strings specifying the output to be read
	datapath -- string for path to hdf5 data i.e. /path/to/hdf5
	name -- string name of type of data i.e. Fluid, RBC, or PLT
	time -- integer for timestep of data to be read in
	Nrunprocs -- integer denoting N cores data was created with
	Nreadprocs -- integer denoting N cores to read data in with 
	"""

	print("Reading ",name," ",time," Nrunprocs ",Nrunprocs," Nreadprocs",Nreadprocs) 

	init_count =0 

	#If output data was created with more then 1 core
	if Nrunprocs >1:
		#Split reading on N readprocs using multiprocessing package
		#Create a pool of worker processes
		pool = Pool(Nreadprocs)
		iter_procs = (i for i in range(Nrunprocs))
		#Prepare the function for pool.map
		read_function = partial(read_hdf5_atomicblock,outputs,datapath,name,time)
		#Read in parallel (mostly benefitioal for restructure fluid data)
		datablocks = np.array(pool.map(read_function,iter_procs))
		pool.close() # Make sure no more processes are submitted to pool
		pool.join() # Wait for the worker processes to finish

		#rejoin all data into one domain from each block via np.concatenate
		for datablock in datablocks:
			if init_count == 0:
				data = datablock
				init_count += 1
			else:
				for key in datablock.keys():
					if len(datablock[key]) >0: #Make sure data is in this block
						data[key] = np.concatenate((data[key],datablock[key]),axis=0)

	#If data was created with a single core just read it in directly
	elif Nrunprocs == 1:
		data = read_hdf5_atomicblock(outputs,datapath,name,time,0)

	#Return desired data as a dictionary over entire domain
	return (data)


if __name__ == '__main__':
	#Specification of an example read
	fluid_outputs = ["Velocity","Density"]
	#Location of sample data from git repo
	datapath = os.getcwd() + "/data"
	time =     3000000
	Nreadprocs = 6
	Nrunprocs = 250

	#Example use 
	#compare read times for multi process vs serial process
	starttime = timeit.default_timer()
	fluiddata = open_hdf5_files(outputs=fluid_outputs,
		datapath=datapath,
		name="Fluid",
		time = time,
		Nrunprocs=Nrunprocs,
		Nreadprocs=Nreadprocs)
	fluidmulti = starttime - timeit.default_timer()

	starttime = timeit.default_timer()
	fluiddata = open_hdf5_files(outputs=fluid_outputs,
		datapath=datapath,
		name="Fluid",
		time = time,
		Nrunprocs=Nrunprocs,
		Nreadprocs=1)
	fluidserial = starttime - timeit.default_timer()

	print("Multiprocessing with ",Nreadprocs," cores ")
	print("Fluid speed up is ",round(fluidserial/fluidmulti,2),"x faster")



	##Uncomment to read in RBC data 

	#rbc_outputs = ["Position","Link force","Cell Id"]
	#rbcdata = open_hdf5_files(outputs=rbc_outputs,
	#	datapath=datapath,
	#	name="RBC_HO",
	#	time = time,
	#	Nrunprocs=Nrunprocs,
	#	Nreadprocs=Nreadprocs)












