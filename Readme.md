`spectrofun` is a python package and CLI for manipulating spectral audio data in a visually intuitive way. If you've seen a spectrogram before, you can understand what you are doing, In essence it's a ring modulator implemented in a weird way. Because it is implement using FFT it does not have limitations on frequency range. It performs great with frequencies below 20hz unlike live ring modulation implementations,


## Functions
You can use spectrofun to frequency shift up or down:

`$ python spect_cli.py infile outfile -f 0.5` shifts the frequencies up by 50% of the frequency range.

You can also invert the frequencies so bass sounds are treble and treble is bass.

`$ python spect_cli.py infile outfile -i`

Finally, you can mix and match these as your heart desires:

`$ python spect_cli.py infile outfile -i -f 0.1 -i -f 0.3` This inverts the frequencies, shifts it all by 10%, inverts it again, and shift it up 30%.


## Limitations
Currently only mono wav files are supported.


## What These Terms Mean in DSP 

I found this write up which explains most of what this tool does: https://archive.is/mRcEy 

At the moment spectrofun basically does ring modulation and frequency shifting, but the behavior is a bit different than what naturally occurs with the signal processing techniques. This tool directly manipulates the spectral data matrix resultinf from FFT rather than DSP techniques more often used in live audio.


/* 


If you set nyq=true then frequency shift actually functions as a ring modulator instead. Because they are both the same thing. But shift usies only the "analytic signal". Analytic signal is the positive portion only.

Invert uses only the non-analytci real signal but treats it as real. 

*/

### TODO
- set up CLI with setuptools and installer / requirements.txt ( https://stackoverflow.com/questions/61168583/simplest-way-to-make-a-python-module-usable-as-a-cli-tool )

- fix bug where if I invert twice there centerpoint of the audio is silent, and it slowly gets quiet and then louder.
- make stereo support
- use ffmpeg to autoconvert filetypes.
- automatic logging of transofrmation type for each file name.
- automatic iterate filenames option in export,.
- nyq and imag options in the cli 
- fix any other fft related bugs
	-cli support for values that are not decimals, like fixed number shift instead o fpercentage. (it give serror for >=1 currently.)
	-- (i think i already implemented this somewher eits just not a defaulyt r something)
- My tool doesn't measure anything in hz, it is just measured in the number of bands in the FFT data or the percentage of the signal. (TBH using hz would be more conveninet perhaps.)

#### Done
- now that cli interface is set up, make it so it actually processes stuff 
- add in some type of copyright or whatever so I can upload it to github
