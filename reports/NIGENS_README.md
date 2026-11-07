***N***eural ***I***nformation processing group ***GEN***eral ***S***ounds database (**NIGENS**)
===============================================================================================

NIGENS is a database provided for sound-related modeling in the field of computational auditory scene analysis, particularly for sound event detection, that has emerged from the Two!Ears project [1].

It contains 1017 wav files of various lengths (between 1s and 5mins), in total comprising 4h:46m of sound material. Mostly, sounds are provided with 32-bit precision and 44100 Hz sampling rate. The files contain sound events in isolation, i.e. without superposition of ambient or other foreground sources.     

Fourteen distinct sound classes are included: alarm, crying baby, crash, barking dog, running engine, burning fire, footsteps, knocking on door, female and male speech, female and male scream, ringing phone, piano. Additionally, there is the general (“anything else”) class. Care has been taken to select sound classes representing different features, like noise-like or pronounced, discrete or continuous.

The general class is a pool of sound events different than the 14 distuingished target sound classes, containing as heterogeneous sounds as possible (303 in total). For example, it includes nature sounds such as wind, rain, or animals, sounds from human-made environments such as honks, doors, or guns, as well as human sounds like coughs. These sounds are intended both as ``disturbance'' sound events (superposing) and as counterexamples to target sound classes.

Wav files are accompanied by annotation (.txt) files that include perceptual on- and offset times of the file's sound events. 

In [2], we have developed and analyzed a robust binaural sound event detection training scheme using NIGENS.

[1] Raake, A., et al (2014). *Two!Ears -- Integral interactive model of auditory perception and experience*.  
[2] Trowitzsch, Ivo, et al (2017). *Robust detection of environmental sounds in binaural auditory scenes*, IEEE/ACM Transactions on Audio, Speech, and Language Processing 25(6), 1344-1356.



## File Lists

`all.flist` is a text file containing a list of all included sounds.  
`NIGENS_8-foldSplit_fold<x>.flist` are lists of eight disjunct folds of equal size.  
The same file lists exist as versions without TIMIT files included.


## Event On- and Offset Times

All wav files are accompanied by an annotation (.txt) file that includes on- and offset times of that file's sound events. 

The "general" wav files do not come with on- and offset time annotations, since these files do not constitute any coherent sound class (in the contrary, by design) and are not intended to be positive examples for classifiers.


## Speech sound files 

Speech sounds were compiled from the GRID [3] and TIMIT [4] copora. The latter one unfortunately is attached with a restrictive license, preventing us from redistributing the respective files. The associated event on- and offset time annotations created by us however we don't remove, and at the end of the document we list the TIMIT files used by us, leaving you with the possibility to get hold of these files yourselves.

[3] Cooke, Martin, et al. *An audio-visual corpus for speech perception and automatic speech recognition*. The Journal of the Acoustical Society of America 120.5 (2006): 2421-2424.  
[4] Garofolo, John S., et al. *TIMIT Acoustic-Phonetic Continuous Speech Corpus.* LDC93S1. Web Download. Philadelphia: Linguistic Data Consortium, 1993


## Credits

The NIGENS database was developed by Ivo Trowitzsch, Jalil Taghia and Youssef Kashef from Technische Universität Berlin.


## License

This dataset is released under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode) license.

If you use this data set, please cite as:

Ivo Trowitzsch, Jalil Taghia, Youssef Kashef, and Klaus Obermayer (2019). *The NIGENS general sound events database*. Technische Universität Berlin, Tech. Rep. arXiv:1902.08314 [cs.SD]  


## Funding

The collection of this dataset was partly funded from the European Union’s Seventh Framework
Programme for research, technological development and demonstration under grant
agreement no 618075.


## Attribution

Other than the files listed below, all sounds were acquired from and kindly granted redistribution for research under above license by [![StockMusic.com](stockmusic-32c523.png)](https://stockmusic.sourceaudio.com). These files have been watermarked to enable misuse detection. Please comply with the license.

Speech sounds were compiled from the GRID [1] and TIMIT [2] copora. 

The following sounds were downloaded and included from freesound.org under attribution licenses:

SteamPipe 3 GhostBlow C1.wav by Jovica                          -- https://freesound.org/s/6598/     
Musicbox_Silent_Night.WAV by bilwiss                            -- https://freesound.org/s/24725/    
Universal bpm - Arabic-Scaled Choir.wav by Experimental Illness -- https://freesound.org/s/38950/    
246abc.wav by patchen                                           -- https://freesound.org/s/3503/     
Nylon Guitar Loop B-1.wav by digifishmusic                      -- https://freesound.org/s/48254/    
krgwoopse.wav by patchen                                        -- https://freesound.org/s/3507/     
Flamenco Loop2.wav by digifishmusic                             -- https://freesound.org/s/41668/    
LucasGonze-FrogInTheWell-bit8.wav by lucasgonze                 -- https://freesound.org/s/58976/    
defe.wav by patchen                                             -- https://freesound.org/s/3508/     
It Goes Down Stab Melody 97BPM.wav by beatgorilla               -- https://freesound.org/s/73397/    
wffnd.wav by patchen                                            -- https://freesound.org/s/3502/     
cellos three chords.wav by jus                                  -- https://freesound.org/s/39557/    
Savanna stomp groove.wav by jobro                               -- https://freesound.org/s/76731/    
110 acoustic grunge.wav by Thirsk                               -- https://freesound.org/s/120980    
mark_sitar_01.wav by mannhawks                                  -- https://freesound.org/s/147895    
percussion choir final 05.wav by klankbeeld                     -- https://freesound.org/s/171916    
double bass test1.wav by wescwave                               -- https://freesound.org/s/169885    
cinematic deep bass rumble.wav by ERH                           -- https://freesound.org/s/34012/    
loop02.wav by Zajo                                              -- https://freesound.org/s/4852/     
Funk Loop 1.wav by beatgorilla                                  -- https://freesound.org/s/20442/    
harp.wav by pryght one                                          -- https://freesound.org/s/27130/    
Phat_bass_line.wav by Flick3r                                   -- https://freesound.org/s/48489/    
Cold_3.wav by Flick3r                                           -- https://freesound.org/s/48456/    
Low D.wav by Kyster                                             -- https://freesound.org/s/117720    
Piano Discord 1 by Aiwha                                        -- https://freesound.org/s/196103    
Piano Note 1 by Aiwha                                           -- https://freesound.org/s/196101    
PoppyRapLoop.wav by L0RdMastere0                                -- https://freesound.org/s/186664    
Alto sax 90bpm Cm 4 solo.wav by simondsouza                     -- https://freesound.org/s/187099    
huboftheuniverseprod__thegutbezahlscream.wav                    -- https://freesound.org/s/118521    
sironboy__woman-scream.wav                                      -- https://freesound.org/s/132106    
arjanvanhoorn0181__young-girl-screaming-1.wav                   -- https://freesound.org/s/139086    
arjanvanhoorn0181__young-girl-screaming-2.wav                   -- https://freesound.org/s/139087    
pitx__grito-08.wav                                              -- https://freesound.org/s/15922     
11linda__quick-female-scream.wav                                -- https://freesound.org/s/168725    
11linda__evil-dark-painful.wav                                  -- https://freesound.org/s/168814    
missozzy__female-scream-01.wav                                  -- https://freesound.org/s/169810    
missozzy__female-scream-02.wav                                  -- https://freesound.org/s/169811    
sangtao__shout.wav                                              -- https://freesound.org/s/189339    
curly-hikari-94__scream-girl1.wav                               -- https://freesound.org/s/219650    
saraonsins__high-pitched-scream.wav                             -- https://freesound.org/s/219663    
annatabernero__19-yeard-old-girl-shouting.wav                   -- https://freesound.org/s/219666    
saraonsins__effort-scream.wav                                   -- https://freesound.org/s/219815    
saraonsins__horror-scream.wav                                   -- https://freesound.org/s/219817    
claudia-clim__scream.wav                                        -- https://freesound.org/s/219866    
polmdx__old-woman-scream.wav                                    -- https://freesound.org/s/219868    
ellenskellington__oneshout.wav                                  -- https://freesound.org/s/220307    
ellenskellington__womanshout.wav                                -- https://freesound.org/s/220312    
ericsueiro__woman-yelling-high-pitched-2.wav                    -- https://freesound.org/s/220822    
alberttrepat__female-scared-scream.wav                          -- https://freesound.org/s/222531    
queen-westeros__woman-scream.wav                                -- https://freesound.org/s/222545    
mariallinas__female-scream-1.wav                                -- https://freesound.org/s/222646    
mariallinas__scream.wav                                         -- https://freesound.org/s/222650    
xtrgamr__nooooaaaah-01.wav                                      -- https://freesound.org/s/240223    
reitanna__gasp-oooh.wav                                         -- https://freesound.org/s/242607    
reitanna__another-surprised-oooh.wav                            -- https://freesound.org/s/242644    
archeos__womanscream.wav                                        -- https://freesound.org/s/261419    
hoerspielwerkstatt-hef__child-scream2.wav                       -- https://freesound.org/s/273153    
11linda__female-screaming-for-help.wav                          -- https://freesound.org/s/326893    
hali-pinson__scream-5.wav                                       -- https://freesound.org/s/328000    
hali-pinson__scream-4.wav                                       -- https://freesound.org/s/328001    
hali-pinson__scream-2.wav                                       -- https://freesound.org/s/328002    
hali-pinson__scream-1.wav                                       -- https://freesound.org/s/328003    
montblanccandies__crazy-screaming.wav                           -- https://freesound.org/s/332044    
analogchill__scream.wav                                         -- https://freesound.org/s/35716     
freqman__psycho-scream-1.wav                                    -- https://freesound.org/s/42847     
freqman__psycho-screams-3.wav                                   -- https://freesound.org/s/42849     
freesound__000-testing-ariana-scream.wav                        -- https://freesound.org/s/50156     
juskiddink__vocal-joy-and-excitement2.wav                       -- https://freesound.org/s/58585     
hellkt__nooo-without-effects.wav                                -- https://freesound.org/s/80922     
lithe-fider__fl-scream-1.wav                                    -- https://freesound.org/s/9703      
thanvannispen__schreeuw1clean.wav                               -- https://freesound.org/s/125127    
skytheguy__victim.wav                                           -- https://freesound.org/s/135758    
tarrei__desperate-scream.wav                                    -- https://freesound.org/s/141242    
ecfike__scream-1.wav                                            -- https://freesound.org/s/154583    
dinsfire__male-voice-screaming-loudly.wav                       -- https://freesound.org/s/169628    
jorickhoofd__sad-cry-7.wav                                      -- https://freesound.org/s/180314    
jorickhoofd__scream-noooh.wav                                   -- https://freesound.org/s/180350    
annatabernero__young-male-scream.wav                            -- https://freesound.org/s/219654    
saraonsins__man-s-scream.wav                                    -- https://freesound.org/s/219814    
adriancalzon__scream-01.wav                                     -- https://freesound.org/s/219858    
acehell__crit-marc.wav                                          -- https://freesound.org/s/219863    
polmdx__high-tone-scream.wav                                    -- https://freesound.org/s/219869    
polmdx__high-pitch-horror-scream.wav                            -- https://freesound.org/s/219871    
ellenskellington__manshout.wav                                  -- https://freesound.org/s/220311    
adriancalzon__scream-05.wav                                     -- https://freesound.org/s/220623    
alberttrepat__male-screaming.wav                                -- https://freesound.org/s/222530    
alberttrepat__male-screaming-out-loud.wav                       -- https://freesound.org/s/222535    
mariallinas__shout.wav                                          -- https://freesound.org/s/222649    
paltsug__wahha-yanga-ripi-a-raja-bip-gonog.wav                  -- https://freesound.org/s/238043    
kwahmah-02__wild-scream.wav                                     -- https://freesound.org/s/245312    
islandan__male-scream.wav                                       -- https://freesound.org/s/267480    
lodovico__terrifying-scream.wav                                 -- https://freesound.org/s/275170    
unclesigmund__ahhh.wav                                          -- https://freesound.org/s/30988     
malexmedia__malexmedia-man-scream.wav                           -- https://freesound.org/s/31818     
nicktermer__scream.wav                                          -- https://freesound.org/s/320247    
liamg-sfx__battle-cry-6.wav                                     -- https://freesound.org/s/323084    
gutturalgutfuck__pissed-off-screams.wav                         -- https://freesound.org/s/330673    
chrillz3r__scream-frightened.wav                                -- https://freesound.org/s/336537    
vtownpunks__screaming-1.wav                                     -- https://freesound.org/s/63632     
patopurific__scream-and-death.wav                               -- https://freesound.org/s/84351     
creativeheroes__helmutscream.wav                                -- https://freesound.org/s/84352     


## TIMIT files

Used TIMIT female speech wav files:   
timit\train\dr1\fcjf0\si1027.wav     
timit\train\dr1\fcjf0\si1657.wav     
timit\train\dr1\fcjf0\si648.wav      
timit\train\dr1\fcjf0\sx127.wav      
timit\train\dr1\fcjf0\sx217.wav      
timit\train\dr1\fcjf0\sx37.wav       
timit\train\dr1\fcjf0\sx397.wav      
timit\train\dr1\fdaw0\si2036.wav     
timit\train\dr1\fdaw0\sx146.wav      
timit\train\dr1\fdaw0\sx236.wav      
timit\train\dr1\fdaw0\sx326.wav      
timit\train\dr1\fdaw0\sx56.wav       
timit\train\dr1\fjsp0\si1434.wav     
timit\train\dr1\fjsp0\si1763.wav     
timit\train\dr1\fjsp0\si804.wav      
timit\train\dr1\fjsp0\sx264.wav      
timit\train\dr1\fjsp0\sx354.wav      
timit\train\dr1\fjsp0\sx444.wav      
timit\train\dr1\fjsp0\sx84.wav       
timit\train\dr1\fmem0\si2007.wav     
timit\train\dr1\fmem0\si747.wav      
timit\train\dr1\fmem0\sx117.wav      
timit\train\dr1\fmem0\sx207.wav      
timit\train\dr1\fmem0\sx297.wav      
timit\train\dr1\fmem0\sx333.wav      
timit\train\dr1\fsjk1\sa1.wav        
timit\train\dr1\fsjk1\sa2.wav        
timit\train\dr1\fsjk1\si1025.wav     
timit\train\dr1\fsjk1\si2285.wav     
timit\train\dr1\fsjk1\si696.wav      
timit\train\dr1\fsjk1\sx125.wav      
timit\train\dr1\fsjk1\sx215.wav      
timit\train\dr1\fsjk1\sx35.wav       
timit\train\dr1\fsma0\si1621.wav     
timit\train\dr1\fsma0\si2251.wav     
timit\train\dr1\fsma0\si991.wav      
timit\train\dr1\fsma0\sx271.wav      
timit\train\dr1\fsma0\sx361.wav      
timit\train\dr1\fsma0\sx451.wav      
timit\train\dr1\fsma0\sx91.wav       
timit\train\dr1\ftbr0\si2181.wav     
timit\train\dr1\ftbr0\si921.wav      
timit\train\dr1\ftbr0\sx111.wav      
timit\train\dr1\ftbr0\sx21.wav       
timit\train\dr1\ftbr0\sx291.wav      
timit\train\dr1\ftbr0\sx381.wav      
timit\train\dr1\fvfb0\si1032.wav     
timit\train\dr1\fvfb0\si1510.wav     
timit\train\dr1\fvfb0\si2292.wav     
timit\train\dr1\fvfb0\sx132.wav      
timit\train\dr1\fvfb0\sx222.wav      
timit\train\dr1\fvfb0\sx312.wav      
timit\train\dr1\fvfb0\sx42.wav       
timit\train\dr1\fvmh0\si1466.wav     
timit\train\dr1\fvmh0\si2096.wav     
timit\train\dr1\fvmh0\si836.wav      
timit\train\dr1\fvmh0\sx116.wav      
timit\train\dr1\fvmh0\sx26.wav       
timit\train\dr1\fvmh0\sx296.wav      
timit\train\dr1\fvmh0\sx386.wav      

Used TIMIT male speech wav files:  
timit\train\dr1\medr0\si1374.wav   
timit\train\dr1\medr0\si744.wav    
timit\train\dr1\medr0\sx114.wav    
timit\train\dr1\medr0\sx204.wav    
timit\train\dr1\medr0\sx24.wav     
timit\train\dr1\medr0\sx294.wav    
timit\train\dr1\medr0\sx384.wav    
timit\train\dr1\mjeb1\si2097.wav   
timit\train\dr1\mjeb1\si837.wav    
timit\train\dr1\mjeb1\sx117.wav    
timit\train\dr1\mjeb1\sx207.wav    
timit\train\dr1\mjeb1\sx27.wav     
timit\train\dr1\mjeb1\sx297.wav    
timit\train\dr1\mjeb1\sx387.wav    
timit\train\dr1\mjwt0\si1291.wav   
timit\train\dr1\mjwt0\si1381.wav   
timit\train\dr1\mjwt0\si751.wav    
timit\train\dr1\mjwt0\sx121.wav    
timit\train\dr1\mjwt0\sx211.wav    
timit\train\dr1\mjwt0\sx301.wav    
timit\train\dr1\mjwt0\sx31.wav     
timit\train\dr1\mjwt0\sx391.wav    
timit\train\dr1\mmgg0\sa1.wav      
timit\train\dr1\mmgg0\sa2.wav      
timit\train\dr1\mmgg0\si1079.wav   
timit\train\dr1\mmgg0\si2339.wav   
timit\train\dr1\mmgg0\sx179.wav    
timit\train\dr1\mmgg0\sx269.wav    
timit\train\dr1\mmgg0\sx359.wav    
timit\train\dr1\mmgg0\sx449.wav    
timit\train\dr1\mmgg0\sx89.wav     
timit\train\dr1\mmrp0\si2034.wav   
timit\train\dr1\mmrp0\si717.wav    
timit\train\dr1\mmrp0\si774.wav    
timit\train\dr1\mmrp0\sx144.wav    
timit\train\dr1\mmrp0\sx324.wav    
timit\train\dr1\mmrp0\sx414.wav    
timit\train\dr1\mmrp0\sx54.wav     
timit\train\dr1\mpgr0\si2040.wav   
timit\train\dr1\mpgr0\si780.wav    
timit\train\dr1\mpgr0\sx420.wav    
timit\train\dr1\mpsw0\si1067.wav   
timit\train\dr1\mpsw0\si2327.wav   
timit\train\dr1\mpsw0\sx167.wav    
timit\train\dr1\mpsw0\sx257.wav    
timit\train\dr1\mpsw0\sx437.wav    
timit\train\dr1\mpsw0\sx77.wav     
timit\train\dr1\mrai0\si2052.wav   
timit\train\dr1\mrai0\si792.wav    
timit\train\dr1\mrai0\sx162.wav    
timit\train\dr1\mrai0\sx252.wav    
timit\train\dr1\mrai0\sx342.wav    
timit\train\dr1\mrai0\sx432.wav    
timit\train\dr1\mrai0\sx72.wav     
timit\train\dr1\mrdd0\si1050.wav   
timit\train\dr1\mrdd0\si2310.wav   
timit\train\dr1\mrdd0\sx150.wav    
timit\train\dr1\mrdd0\sx277.wav    
timit\train\dr1\mrdd0\sx330.wav    
timit\train\dr1\mrdd0\sx60.wav     
