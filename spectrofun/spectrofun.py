
#TODO: idea: uncoil/recoil at faster/slower speed to change pitch? (this is probablyw hat audacity already does)
#TODO: find a way to extend the sample: expand the time region to do IDFT on so that arbitarily more sound is generated from the same fourier data
	#actually i tihn there is nothing interesting, it will just be mirrorimage of audio signal.
	#like for some reason these signals end up "periodic" in the 2x range of the sound.


import wave
import numpy as np
from scipy.io.wavfile import write
from scipy.fft import ifft as ifft #I wanted to use my own implementation but i dont wanna need to implement fft. ANd I can just reshape thigns befor converting it.
from scipy.fft import fft as fft


max_int16 = 2**15


##### CONVERSION FUNCTIONS #####

def DFT(x):
    """
    Function to calculate the 
    discrete Fourier Transform 
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = np.dot(e, x) #calculates values for combo of wt and sum over t. So that each item in this is for a corresponding w (frequency)
    
    return X

def IDFT_better(X): #TODO test if it actually works
    """
    Rever DFT"""
        
        
    N = len(X) #todo: can change the length on export by changing this. I think. but it might just slow the sample down. but need to be able to do dot product also.

    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(j * k * n / N)
 
    x = np.dot(e, X)
    
    return np.real(x)/100. 



def IDFT_longer(X): #probably doesnt work
    """
	IDFT but apply it to a longer sample , ideally so we can see what happens as the sinusoids continue onwards
	"""
    X = np.hstack([X,np.zeros(len(X))])    
        
    N = len(X) 
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = -1/np.exp(-2j * np.pi * k * n * 2 / N) #THIS TAKES A LONG TIME. #fIXME
    x = np.dot(e, X)
    
    return np.real(x)/100. 


def IDFT(X):
    """
    Inverse DFT
    """
        
        
    N = len(X) #todo: can change the length on export by changing this. I think. but it might just slow the sample down. but need to be able to do dot product also.

    n = np.arange(N)
    k = n.reshape((N, 1))
    e = -1/np.exp(-2j * np.pi * k * n / N)
 
    x = np.dot(e, X)
    
    return np.real(x)/100. #FIXME im just brute forcing this, not using a formula. I dont think i should have to do this.
    #FIXME: dropping the complex is mapping -1 to 1 sometimes???? idk
    #phase DOES WORK becuase this leftover imaginary is very small.
    #I'm also adding weird harmonics somehow. Like, there are small wavering in the audio which is anothe rpitch. IDK why or how to fix.

def magnitude(X):
	return np.real(X)
def phase(X):
	return np.imag(X)
def normalize(x):
	"""???"""
	return x/max_int16

def to_16b(x):
	"""
	takes ndarray
	make it loud and ready to be 16 bit audio
	"""
	return (max_int16 * (x / max(x))).astype('int')

##### FFT Manipulating Functions #####

def invert_one(X):
	XLi = list(X)
	return np.array(XLi[len(XLi)//2:len(XLi)-1] + XLi[0:len(XLi)//2])

def invert_two(X): #OLD
#Old. / Doesn't work. It reverses the audio which was not intended effect.
	XLi = list(X)
	return np.array(XLi[::-1]) #reversed() is probably faster but it returns as iterator and you cant call stuff on it. lazy solution for now.

def offset_old(X,amount:int):
	"""Expected: shift samples up or down by some amount.basically offsetting it on the visual map
	Actual: It shifts the "fourier transform" up or down. So you end up witht he mirrored version that's "hidden underneath" being revealed in a certain spot.
	"""
	XLi = list(X)
	end=len(XLi) - 1
	start = 0
	return np.array(XLi[end-amount:end] + XLi[start:end-amount])#todo


def offset_buggybutcool(X,amount:int, nyq=True): #todo: add option pre-nyquist, post-nyquist, or all i guess?
#this one, i accidentally appended to many samples..
#In effect this is adding stuff to the top, but not shifting anything. It "compresses"the original sample and adds more of a scond
#sample on top of that in freqs, while also stretching / slowing down (making entire thing slightly deeper in process)
#Sounds really nice tho!

#FIXED:weird things happened bc I didn't check that the pre-nyquist part was longer than "amount". Somehow this made
#the post nyquist part leak into pre-nyquist while also adding on the entire post-nyquist part, making it longer.

	"""Expected: shift samples up or down by some amount.basically offsetting it on the visual map
	Actual: It shifts the "fourier transform" up or down. So you end up witht he mirrored version that's "hidden underneath" being revealed in a certain spot.
	"""

	#bug: when the amount is around 0.63 of the length something weird happens and the output becomes longer than input???
	XLi = list(X)
	pre_nyq = len(XLi)//2
	post_nyq = pre_nyq + len(XLi)%2
	start = 0
	true_end = len(XLi) - 1
	if nyq: #only shift first half of audio.
		end = pre_nyq
		return np.array(XLi[end-amount:end] + XLi[start:end-amount] + XLi[post_nyq:true_end])#todo
	else: #shift entire audio (ends up with symmetric effect)
		#end=len(XLi) - 1
		return np.array(XLi[true_end-amount:true_end] + XLi[start:true_end-amount])#todo



def offset(X,amount:float, nyq=True, imag = True): #todo: add option pre-nyquist, post-nyquist, or all ig uess?
#TODO: finish debugging by comparing an original file to an offset amount=0 file. Theres something fishy going on still
	"""shift samples up or down by some amount.basically offsetting it on the visual map
	Actual: It shifts a second copy up or down, overlapping the original... (need to look if this is still true)

	numbe rin (0,1) treated as percentage.
	otherwise treated as discrete number of units in the offset of the spectral values. (discrete frequency steps)
	"""

	XLi = list(X)
	pre_nyq = len(XLi)//2
	post_nyq = pre_nyq + len(XLi)%2 #Does nothing due to how arrays work but don't wan tto delete quite yet
	start = 0
	true_end = len(XLi)
	assert true_end > amount , "Shift amount cannot exceed audio length of %d" % true_end

 
	if -1 < amount < 0:
		amount = 1 + amount
	if 0 < amount < 1:
		#convert to correct value.
		amount *= pre_nyq if nyq else true_end
		amount = int(amount)

	assert amount != 0 , "Shift amount must be non-zero.."

	if nyq: #only shift first half of audio. ("analytic")
	# shift the pre-nyq part. Also shift the post-nyq part in a mirrored way. (same as ring modulation I think)
		end = pre_nyq
		assert end > amount, "Shift amount cannot be longer than pre-nyquist audio. Set nyq=False or lower amount below %d" % end
		#left half is pre-nyq, and right half is post-nyq. post-nyq does same thing but mirrored.
		#if you don't mirror it you end up with overlap of two separate signals. IDk if they hold separate info or not.
		result = np.array(XLi[end-amount:end] + XLi[start:end-amount] + XLi[pre_nyq+amount:true_end] + XLi[pre_nyq:pre_nyq+amount])
	else: #shift entire audio (ends up with symmetric effect)
		#end=len(XLi) - 1
		result =  np.array(XLi[true_end-amount:true_end] + XLi[start:true_end-amount])#UNTESTED #TODO

	return result
	


##### Class #####

class SpectroAudio():
	"""
	object that has all the modifying functions and takes infile and out file.
	"""
	def __init__(self, infile,ofile=None):
		self.ifile = infile 
		self.ofile = ofile
		self.data = wave.open(self.ifile)
		self.samples = self.data.getnframes()
		self.audio = self.data.readframes(self.samples)
		self.paudio, self.spectral, self.N, self.n = self._preprocess()

		#there are used for offset.. IDK if i need them here really tbh.
		self.nyq_size = len(self.spectral)//2#aka ore_nyq or end previously
		self.size = len(self.spectral) #aka true_end previous

		self.srate = 48000
		self.max_int16 = 2**15

	def _preprocess(self, audio = None):
		"""
		normalize audio, make it a float,  and generate spectral data

		Returns normalized audio as a np array, fft matrix np array, size of matrix & np.arange(N) 
		"""
		if audio is None:
			audio = self.audio

		# Convert buffer to float32 using NumPy                                                                                 
		audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
		audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

		# Normalise float32 array so that values are between -1.0 and +1.0                                                      
		audio_normalised = audio_as_np_float32 / max_int16

		X = fft(audio_normalised)
		# calculate the frequency
		N = len(X)

		return audio_normalised, X, N, np.arange(N) 


	def export(self, ofile=None,srate = None, spec = None):
		if ofile is None:
			ofile = self.ofile
		assert ofile is not None #should throw typeerror missing one position argument, UNTESTED
		if srate is None:
			srate = self.srate
		if spec is None:
			spec = self.spectral 

		out_data = to_16b(ifft(spec)).astype(np.int16)
		write(ofile, srate, out_data)
		return 

	def offset(self,amount:float, nyq=True, imag = True):
		self.spectral = offset(self.spectral,amount,nyq,imag)
	def invert(self):
		self.spectral = invert_one(self.spectral)