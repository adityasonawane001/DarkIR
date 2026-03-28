# DarkIR Project - YouTube Video Demo Script (15 Minutes)

**Goal:** ~10 minutes explaining the paper and your 4 novel contributions + ~5 minutes showcasing a live, working Gradio demo.
**Pro Tip for Recording:** Screen-record using OBS Studio or Zoom. Keep the downloaded `DarkIR_Official_Paper.pdf` open in a PDF viewer where you can use a highlighter tool to physically mark the paper while you talk.

---

## Part 1: Base Paper Explanation (0:00 - 5:00)

**[Visual on Screen]** Open the `DarkIR_Official_Paper.pdf` to the Title and Abstract.

**Speaker (Kumar, Aditya, or Roshan):**
> "Hello everyone, welcome to our Deep Learning course project presentation. Today we are presenting our work on **DarkIR: Robust Low-Light Image Restoration**. This paper was published very recently at CVPR. 
> 
> The core problem that this paper tackles is that when you take a photo in severely low-light environments, you don't just get a dark image—you get a coupled degradation of darkness, blur from shaky camera hands, and heavy sensor noise."

**[Visual on Screen]** Scroll down to the Introduction block and **Highlight Equation 1** ($y = \gamma(x \otimes k) + n$).

**Speaker:**
> "To understand the complexity, look at this formulation. Traditional methods try to fix these sequentially—first denoise, then brighten, then deblur. But as seen in Equation 1, the clean image $x$ is simultaneously blurred by $k$, altered by the camera response function $\gamma$, and corrupted by noise $n$. Trying to fix one without the others leads to artifacts. DarkIR proposes to solve them together using a dual-domain network."

**[Visual on Screen]** Scroll down to the **Architecture Diagram (Figure 2)** and highlight the two distinct paths (Spatial and Frequency).

**Speaker:**
> "The base DarkIR model avoids computationally expensive Transformers. Instead, it relies on an efficient CNN with two primary processing streams:
> 1. A **Spatial Domain Path** using dilated convolutions to fix local corruptions like noise and blur.
> 2. A **Frequency Domain Path** using Fast Fourier Transforms (FFTs) to capture global illumination and structural color adjustments.
> This makes the baseline incredibly fast and lightweight. However, during our reproduction, we found several areas where we could push the performance and stability even further."

---

## Part 2: Our Novel Contributions (5:00 - 10:00)

**[Visual on Screen]** Switch to your PPT presentation (specifically the **Contributions flow diagram** we generated in LaTeX).

**Speaker:**
> "While reproducing this architecture, we implemented four distinct novel contributions to improve visual quality and model learning."

**Contribution 1: Phase-Aware Frequency Modulation**
> "Our first contribution is in the Frequency Path. The standard DarkIR primarily enhances the magnitude of the spectrum and leaves the phase mostly untouched. However, phase contains critical structural information. We introduced a completely new learnable phase-weighting branch using a $1 \times 1$ convolution and strictly bounded epsilon mapping. This allows the network to intentionally shift phase geometry, leading to crisper edges."

**Contribution 2: Content-Adaptive Receptive Fields**
> "Our second contribution targets the Spatial Path. The base paper hardcodes dilated convolutions to merge local and global textures statically. We implemented a dynamic spatial block constraint using Global Pooling and Softmax. Now, the model actively 'votes' on whether it needs to rely on local $3 \times 3$ patterns for noise, or wide $9 \times 9$ patterns to repair large blurs, adapting on a per-image basis."

**Contribution 3: Gated Channel Attention Matrix**
> "Third, we upgraded the feature gating mechanism. Instead of a simple element-wise multiplication, we split the feature channels into chunks, learned a powerful attention mask from one half, and gated the second half. This creates a selective routing mechanism that filters out noisy channels before reconstructing the image."

**Contribution 4: Dual-Spectrum Frequency Loss**
> "Finally, we proposed a Novel Frequency Loss. Traditional training only penalizes L1 combinations at the pixel level. We wrote a custom loss function that enforces L1 consistency not just on the spatial output, but explicitly on the Amplitude and Phase maps of the Fourier transform. This was notoriously unstable to train, so we mathematically bounded it using Safe Complex Math to prevent NaNs on our GPU."

---

## Part 3: Working Code Demo (10:00 - 15:00)

**[Visual on Screen]** Switch to VS Code. Show the terminal and the `app_local.py` file.

**Speaker:**
> "For the final part of our presentation, we will demonstrate our model physically running on our local hardware. Because of our optimizations, we were able to deploy this on a standard Laptop RTX 3050.
> 
> Here in VS Code, you can see our custom inference script `app_local.py`. It dynamically loads our trained Distributed Data Parallel checkpoints, handles high-resolution tensor downscaling automatically, and wraps everything into a Gradio Web UI."

**[Action]** Run `python app_local.py` in the terminal. Wait for the `http://127.0.0.1:7861` link to appear.

**Speaker:**
> "Our server is live on localhost. Let's open it in the browser."

**[Visual on Screen]** Open browser to the Gradio interface. 

**[Action]** Drag and drop your extremely dark test image (`qualis_realblur_night.jpg` / `demo.jpg`) into the input box.

**Speaker:**
> "We are submitting a severely underexposed, noisy, and blurred image. The UI gives us a 'Visibility Boost' slider, which applies our custom non-linear gamma curve to help map the lowest light pixels properly before feeding it into the tensors.
> 
> Let's hit submit."

**[Action]** Click Submit. Wait the ~2 seconds for it to process. 

**Speaker:**
> "As you can see, in under just a few seconds, the model perfectly reconstructs the lighting, stabilizes the color phase to avoid weird green/purple tints, and aggressively filters out the sensor noise without blurring the face. We successfully mapped the global illumination via the frequency domain and the local textures via our adaptive spatial dilations."

**[Visual on Screen]** Show the side-by-side comparison on Gradio.

**Speaker:**
> "You can see the final inference logs here reflecting the exact device footprint. Our modifications improved structural integrity and allowed interactive, near real-time performance. 
> 
> Thank you for watching!"