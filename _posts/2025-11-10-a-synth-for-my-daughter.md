---
layout: post
title: "A Synth for My Daughter"
date: 2024-01-15
category: Hardware
excerpt: "How I built a portable synthesizer for my three year old daughter."
tags: [hardware, software, iot, raspberry-pi, nodejs]
---

{% assign base = '/assets/posts/2025-11-10-a-synth-for-my-daughter' | relative_url %}


<figure> 
  <img style="max-width: 600px" src="{{ base }}/prototype-enclosure.jpeg" alt="Slider layout">
</figure>


<p class="tldr">TLDR: I've made my daughter a synthesiser for her third birthday. Four sliders control four notes which play one after the other, looping endlessly. Pushing a slider up raises the pitch of a note at a given step and pushing it down lowers it. It has a few basic functions that allow the user to control the volume, tempo, pitch, scale and sound and comes with a selection of patterns to choose from. It also has a screen for visual feedback and displaying a cute, dancing panda bear. Here it is in action:</p>

## Inspiration

For my daughter's first birthday, she was gifted a ['Montesorri Activity Board'](https://www.amazon.co.uk/Montessori-Activity-Sensory-Learning-Birthday). About the size of a paperback, it presents itself as a wooden box with various knobs and switches that control a constellation of multicoloured LEDs. It reminded of a the control surface of a synthesizer and I thought that if it could produce a few simple sounds it might make it more interesting, and possibly educational, for a child who uses it. This gave me the idea of building an accessible electronic instrument. A sequencer - that is, a controller that plays user-chosen notes one after another at regular time intervals - seemed like a natural interface since music can play without constant effort on part of the user while retaining enough creative potential for it to remain interesting. I thought sliders would provide an intuitive way of controlling the pitch of the notes since their position maps to the language that we use when talking about pitch i.e. up, down, high, low.

It wasn't until her birthday the following year when we gifted her a Yoto Mini player that I resolved to actually build the thing. The Yoto Mini is a small portable audio player that allows children to play music and stories by inserting the NFC-powered cards into its slot. In addition to a large library of story book and music cards you can also purchase blank ones and use their app to connect them to your own uploaded MP3s. It is a beautiful device and simple enough for a toddler to operate. I was moved by my daughter's initial reaction when she first held this little box as it played some of her favourite music and wanted to channel some creative energy into building something fun for her.

## Prototyping

I started the project with a 15 year old Arduino Inventors Kit and no prior hardware experience. I figured that a reasonable place to start was to breadboard a simple MIDI controller for some as-yet-undetermined module that would generate the audio. If I could get some potentiometer readings, map them to 12 discrete values - one for each note in an octave - and turn them somehow into MIDI messages, I would have taken a small step in the right direction. Designing a pretty box to put it in could wait until later.

Reading the potentiometer inputs was easy enough and I then used the [Arduino MIDI](https://github.com/FortySevenEffects/arduino_midi_library) library to generate the MIDI messages. To hear the output I wrote a small Python script that could intercept the generated MIDI signals and forward them on to my MBP's default MIDI device which could then be picked up by Logic Pro. This meant that I could use software instruments to bring the Arduino sequencer to life in the initial stages.

<figure> 
  <img style="max-width: 600px" src="{{ base }}/breadboard-prototype.jpeg" alt="Slider layout">
</figure>

Once I'd got the hang of wiring up potentiometers and rotary encoders, the next step was to move the audio synthesis from Logic to my breadboard. For this I used a little $12.95 [SAM2695 synthesiser module](https://shop.m5stack.com/products/midi-synthesizer-unit-sam2695) with an integrated amplifier and speaker. Its inner workings remain a mystery to me but it does what I need it to and I was happy reduce the amount of time to get a functioning prototype into my daughter's hands.

Next, I added small OLED screen to show provide some visual feedback and character and used the handy [u8g2 graphics library](https://github.com/olikraus/u8g2). This was actually more of a challenge to incorporate that anticipated since the Arduino Nano I was using has so little RAM that storing an entire screen-sized bitmap in memory and writing to the OLED in one go was out of the question. This meant that I had to be more selective with the parts of the screen I was updating. At this point I really should've switched to a much more powerful ESP32 variant which would have simplified the implementation. Updating the screen in patches is expensive and large updates could take sufficiently long that encoder readings would be interrupted resulting in dodgy controls and delayed notes. Nevertheless I persevered and accepted a little bit of MIDI lag at quicker tempos when twiddling a knob.

For coding on-the-go I discovered a fantastic [microcontroller simulator called Wokwi](https://wokwi.com/) that allowed me create a diagram of my circuit and test my implementation without having to lug around my delicate prototype. They have an online simulator that's available for free and a VS Code plugin in their paid version.

<div class="image-gallery cols-2">
<a href="{{ base }}/wokwi.jpeg">
    <img src="{{ base }}/wokwi.jpeg" alt="Woki in VS Code" style="object-position: center;">
  </a>
<a href="{{ base }}/breadboard.jpeg">
    <img src="{{ base }}/breadboard.jpeg" alt="Early breadboard prototype" style=" height: 100%; object-position: center">
  </a>
</div>

With a broadly function circuit it was time to move on to designing an enclosure and assembling a complete version of the synthesiser that my daughter could play with.

## Version 1

The prospect of moving my circuit, however simple, to a PCB seemed daunting so I decided the first version would be hand-wired together using a solderable breadboard. The good: drinking wine and hanging with Romain, who kindly offered to help with the soldering; the bad: when the time finally came to close the two halves of the enclosure, stuffing the rats nest of wires inside ended up putting pressure on a bunch of the delicate soldered joints and breaking them. My daughter could play around a bit with it - enough for me to convince/fool myself that she'd enjoy using it - but it was fragile and I resolved to make an improved version. I also decided that I'd build a few units to give to some friends and wanted something more robust that was quicker and easier to assemble.

<div class="image-gallery cols-2">
<a href="{{ base }}/assembly-1.jpeg">
    <img src="{{ base }}/assembly-1.jpeg" alt="v1 assembly" style=" height: 100%; object-position: center">
  </a>
<a href="{{ base }}/v1.gif">
    <img src="{{ base }}/v1.gif" alt="v1 in action" style="height: 400px; object-position: left;">
  </a>
</div>

I also made a poor decision about the power supply. After learning that energy capacity is a thing and figuring out that a 9V battery has too little to power my components for a reasonable amount of time, I decided to opt for a 4 AA batteries and use the Arduino's built-in voltage converter to provide a steady 5 volts. Something I overlooked, however, is that the Arduino's VIN pin that provides a regulated 5V to the board requires 7-12V input, while my batteries will provide, at best, 6V when new. The board seemed to work OK at this voltage but it still seemed like a bad idea since it would likely lead to flakiness and short battery life when the voltage starts to sag.

Building a new version would allow me to address these shortcomings. I was also keen to make a proper battery compartment.


## Version 2

### PCB

For the follow-up version I decided to pull up my socks and learn some elementary PCB design. Fusion 360 comes bundled with PCB design software, though there are some restrictions in place in the free version that limits the board to two layers and the size of the board to 80mm^2. The goal here was simply to provide a surface on which I could attach the various components and have them all wired up after soldering their pins. It would've been more cost effective to integrate the Arduino's Atmega chip and circuitry directly onto the board, however I wanted to keep the circuit simple and focus on learning the end-to-end flow of designing a PCB and incorporating it into a device. 

For those who are unfamiliar with designing PCBs in Fusion 360, they consist of three 'sheets' - a schematic, the 2D PCB layout containing component footprints and routing (wiring), and a 3D model that you can then incorporate in your enclosure. When you've decided on the components you need, you can go to one of the many online electronics retailers such as Mouser, locate a suitable part, and download some manufacturer-provided files containing the schematic, footprint and 3d model. You can then import them into Fusion 360, drag them into place and wire 'em up. When you're done, you then export the PCB design file and upload this to a manufacture's website, give them $30 and twiddle your thumbs for 5 or so days before receiving it in the post. This last part is a modern miracle.

<div class="image-gallery cols-2">
<a href="{{ base }}/pcb-design.jpeg">
    <img src="{{ base }}/pcb-design.jpeg" alt="PCB design in Fusion 360" style="object-position: center;">
  </a>
<a href="{{ base }}/fistful-of-pcbs.jpeg">
    <img src="{{ base }}/fistful-of-pcbs.jpeg" alt="Manufactured PCBs" style=" height: 100%; object-position: center">
  </a>
</div>

### Enclosure

With the PCB designed and ordered, I moved on to designing the enclosure that would house all the components. I used Fusion 360's CAD tools to model the case, ensuring proper spacing for the sliders, screen, speakers, and battery compartment. Here are some views of the enclosure design and assembly:

<div class="image-gallery cols-2">
<a href="{{ base }}/enclosure-bottom.jpeg">
    <img src="{{ base }}/enclosure-bottom.jpeg" alt="Enclosure bottom view" style="width: 600px; height: 400px; object-position: center; clip-path: inset(50px 0 20px 0);">
  </a>
<a href="{{ base }}/enclosure-battery.jpeg">
    <img src="{{ base }}/enclosure-battery.jpeg" alt="Battery compartment" style="width: 600px; height: 400px; object-position: center; clip-path: inset(50px 0 20px 0);">
  </a>
  
  <a href="{{ base }}/enclosure-pcb.jpeg">
    <img src="{{ base }}/enclosure-pcb.jpeg" alt="PCB fitted in enclosure" style="width: 600px; height: 400px; object-position: center; clip-path: inset(50px 0 20px 0);">
  </a>
  <a href="{{ base }}/enclosure-cad.jpeg">
    <img src="{{ base }}/enclosure-cad.jpeg" alt="Enclosure CAD design" style="width: 600px; height: 400px; object-position: center; clip-path: inset(50px 0 20px 0);">
  </a>
</div>

### Final assembly

<div class="image-gallery cols-2">
<a href="{{ base }}/v2-pcb.jpeg">
    <img src="{{ base }}/v2-pcb.jpeg" alt="Final assembly - PCB in enclosure" style="object-position: center;">
  </a>
<a href="{{ base }}/v2-back-panel-removed.jpeg">
    <img src="{{ base }}/v2-back-panel-removed.jpeg" alt="Final assembly - Back panel" style=" height: 100%; object-position: center">
  </a>
</div>

## Reflections & next steps



---

*This project perfectly embodies the "Bits & Pieces" philosophy - combining various technologies and approaches to create something greater than the sum of its parts. The intersection of hardware hacking, web development, and home automation has been incredibly rewarding.*

*What smart home projects are you working on? I'd love to hear about your experiences in the comments below.* 