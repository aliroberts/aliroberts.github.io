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


TLDR: My daughter is about to turn three and, for her birthday, I've made her a synthesiser. Four sliders control four notes which play one after the other, looping endlessly. Pushing a slider up raises the pitch of a note at a given step and pushing it down lowers it. It has a few basic functions that allow the user to control the volume, tempo, pitch, scale and sound and comes with a selection of patterns to choose from. It also has a screen for visual feedback and displaying a cute, dancing panda bear. Here it is in action:

## Inspiration

When my daughter turned one she was gifted a little wooden busy-board (LINK). It was interesting to watch it pique the interest of the kids and the parents (though mainly the Dads). For the latter I think it presented them with a sort of frustrated joy - joy in the collection of little gizmos and lights that resembled something actually interesting combined with a low-level frustration in the ultimate lack of any depth of the experience. This gave me the vague idea of building an accessible electronic instrument aimed at children of all ages. The idea of using a sequencer as the interface seemed natural since the music can play without constant effort on part of the user while retaining enough creative potential for it to remain interesting. The sliders seemed like an intuitive way of interacting with the notes and allowing the child to develop an intuitive understanding of pitches.

It wasn't until her birthday the following year when we gifted her a Yoto Mini player that I resolved to actually build the thing. For those who are unfamiliar, the Yoto Mini is a small portable audio player that allows children to play music and stories by inserting the NFC-powered cards into its slot. In addition to a large library of story book and music cards you can also purchase blank ones and use their app to connect them to your own uploaded MP3s. It is a beautiful device and simple enough for a toddler to operate. I was moved by my daughter's initial reaction when she first held this little box as it played some of her favourite music and wanted to channel some creative energy into building something fun for her.

## Prototyping

I started the project as a coder and musician with no hardware experience. The path towards the finished product was hazy so I did what I usually do when starting any product and try as best I can to isolate the components that seem relatively self-contained and focus on one of those.  I had an unused 15 year old Arduino Uno Inventors Kit lying around and figured that a reasonable place to start was to breadboard a simple MIDI controller for some as-yet-undetermined module that would generate the audio. If I could get some potentiometer readings, map them to 12 discrete values - one for each note in an octave - and turn them somehow into a MIDI signal, I would have taken a small but significant step.

The https://github.com/FortySevenEffects/arduino_midi_library library made light work of the MIDI part. To hear the output I wrote a small Python script that could intercept the generated MIDI signals and forward them on to my MBP's default MIDI device which could then be picked up by Logic Pro. This meant that I could use software instruments to bring the Arduino sequencer to life in the initial stages.

<figure> 
  <img style="max-width: 600px" src="{{ base }}/breadboard-prototype.jpeg" alt="Slider layout">
</figure>

Once I'd got the hang of wiring up potentiometers and rotary encoders I decided to incorporate the actual synthesiser bit into my prototype. For this I reached for a little [SAM2695 synthesiser module](https://shop.m5stack.com/products/midi-synthesizer-unit-sam2695) I found that came with an integrated amplifier and speaker. The engineer in me did so guiltily since it was (and still is) a black box but I figured that I could eventually isolate this part and do a deep dive and make my own custom version if time and inclination would allow.

I followed this up by adding small OLED screen to show provide some visual feedback and character and used Oli Kraus's excellent (u8g2 graphics library)[https://github.com/olikraus/u8g2]. This was actually more of a challenge to incorporate that anticipated since the Arduino Nano I was using has so little RAM that storing an entire screen-sized bitmap in memory and writing to the OLED in one go was out of the question. This meant that I had to be more selective with the parts of the screen I was updating. At this point I really should've switched to a much more powerful ESP32 variant which would've eliminated these concerns. Updating the screen in patches is expensive and large updates could take sufficiently long that encoder readings would be interrupted resulting in dodgy controls and delayed notes. Nevertheless I persevered and accepted a little bit of MIDI lag at quicker tempos when twiddling a knob.


<figure> 
  <img style="max-width: 600px" src="{{ base }}/wokwi.jpeg" alt="Slider layout">
</figure>

- PHOTO OF EARLY VERSION ('Just another few weeks! Six months later...')

With a broadly function circuit it was time to move on to designing an enclosure and building the first complete version of the synthesiser that my daughter could hold and use.

## Version 1

### Assembly

The prospect of moving my circuit, however simple, to a PCB seemed daunting so I decided the first version would be hand-wired together using a solderable breadboard. What I initially thought was a smart simplification that would enable me to get the thing into my daughter's hands more quickly turned out to be waste of time, however. When the time finally came to close the two halves of the enclosure, stuffing the rats nest of wires inside ended up putting pressure on a bunch of the delicate soldered joints and breaking them. My daughter could play around a bit with it - enough for me to convince/fool myself that she'd enjoy using it -  but it was fragile and I resolved to make an improved version (in the colour of her choice). I also had the vague idea that I'd build a few units for some friends and the thought of soldering 4 or 5 more of them didn't fill me with joy.

<figure> 
  <img style="max-width: 600px" src="{{ base }}/assembly-1.jpeg" alt="Slider layout">
</figure>

### Power

I also made a poor decision about the power supply. After learning that energy capacity is a thing and figuring out that a 9V battery has too little to power my components for a reasonable amount of time, I decided to opt for a 4 AA batteries and use the Arduino's built-in voltage converter to provide a steady 5 volts. Something I overlooked, however, is that the Arduino's VIN pin that provides a regulated 5V to the board requires 7-12V input, while my batteries will provide, at best, 6V when new. The board seemed to work OK at this voltage but it still seemed like a bad idea since it would likely lead to flakiness and short battery life when the voltage starts to sag.

Building a new version would allow me to address these shortcomings. I was also keen to make a proper battery compartment.


- IMAGE OF VERSION 1 "Ooh, I'm blinded by the lights"

## Version 2

<figure> 
  <img style="max-width: 600px" src="{{ base }}/fistful-of-pcbs.jpeg" alt="Slider layout">
</figure>

### PCB

For the follow-up version I decided to pull up my socks and learn some elementary PCB design. Fusion 360 comes bundled with PCB design software, though there are some restrictions in place in the free version that limits the board to two layers and the size of the board to 80mm^2. The goal here was simply to provide a surface on which I could attach the various components and have them all wired up after soldering their pins. It would've been more cost effective to integrate the Arduino's Atmega chip and circuitry directly onto the board, however I wanted to keep the circuit simple and focus on learning the end-to-end flow of designing a PCB and incorporating it into a device. 

For those who are unfamiliar with designing PCBs in Fusion 360, they consist of three 'sheets' - a schematic, the 2D PCB layout containing component footprints and routing (wiring), and a 3D model that you can then incorporate in your enclosure. When you've decided on the components you need, you can go to one of the many online electronics retailers such as Mouser, locate a suitable part, and download some manufacturer-provided files containing the schematic, footprint and 3d model. You can then import them into Fusion 360, drag them into place and wire 'em up. When you're done, you then export the PCB design file and upload this to a manufacture's website, give them $30 and twiddle your thumbs for 5 or so days before receiving it in the post. This last part is a modern miracle.

### Enclosure

With the PCB designed and ordered, I moved on to designing the enclosure that would house all the components. I used Fusion 360's CAD tools to model the case, ensuring proper spacing for the sliders, screen, speakers, and battery compartment. Here are some views of the enclosure design and assembly:

<div class="image-gallery">
  <a href="{{ base }}/enclosure-cad.jpeg">
    <img src="{{ base }}/enclosure-cad.jpeg" alt="Enclosure CAD design">
  </a>
  <a href="{{ base }}/enclosure-pcb.jpeg">
    <img src="{{ base }}/enclosure-pcb.jpeg" alt="PCB fitted in enclosure">
  </a>
  <a href="{{ base }}/enclosure-bottom.jpeg">
    <img src="{{ base }}/enclosure-bottom.jpeg" alt="Enclosure bottom view">
  </a>
  <a href="{{ base }}/enclosure-battery.jpeg">
    <img src="{{ base }}/enclosure-battery.jpeg" alt="Battery compartment">
  </a>
</div>

## Reflections & next steps
---

*This project perfectly embodies the "Bits & Pieces" philosophy - combining various technologies and approaches to create something greater than the sum of its parts. The intersection of hardware hacking, web development, and home automation has been incredibly rewarding.*

*What smart home projects are you working on? I'd love to hear about your experiences in the comments below.* 