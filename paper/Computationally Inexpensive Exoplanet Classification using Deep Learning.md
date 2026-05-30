# Computationally Inexpensive Exoplanet Classification using Deep Learning
By Pierce Nestibo-Oxley

## 1 Introduction

Launched in April 2018, the TESS (Transiting Exoplanet Survey Satellite) mission is a NASA and MIT collaboration that searches for exoplanet transits \[1\], \[2\], slight periodic dimmings in the observed brightness of a star. These dips are caused by the opaque occlusion of an exoplanet around its host star. In simple terms, an exoplanet can line up in such a way that it blocks some of the light from its host star from reaching sensors around Earth. By recording the observed brightness across many time instances, we can construct a light curve for that specific star, and from the curve, find patterns that indicate the presence of an exoplanet candidate. 

Humans are typically quite good at this type of work; we have evolved to be efficient at identifying and comparing patterns, and an exoplanet signal is nothing but a collection of patterns needing to be evaluated. Fairly easily, we can learn to decide between planet candidates and false positives, and often without needing a rigorous mathematical or scientific background in the field. Unfortunately, relying on human judgement comes with a one sizeable drawback: Humans are slow. It can take months of casual practice for a human to become adept at classifying planets, binary systems, and their related, similar false positives. Furthermore, once fully trained, it may take an experienced person several minutes to review and analyze all the information to make a single informed decision. At these speeds, classifying a modest number of possible planet signals — on the order of 100 to 1000 ($10^2$ to $10^3$) — can take days. Given the pure increase in volume of astronomical data in recent years,  it is becoming impractical to rely on human classifications for viable planet candidates. TESS itself, for example, tracks \~200,000 ($2 \times 10^5$) stars as part of its primary catalogue, and observes on the order of 10 million more ($10^7$). With improvements to consumer cameras, measuring equipment, and computing access, it is also easier for amateur astronomers to individually collect large amounts of data. An alternative system, one capable of quickly, accurately, and repeatedly identifying planet candidates, would therefore be highly useful to exoplanet researchers.

For such a modern field in the study of astronomy, the methods behind exoplanet classification leave much to be desired for the amateur astronomer. For 30 years after the confirmation of the first exoplanet in 1992, most analysis was done manually, with a scientist individually overseeing every single potential candidate. In recent years, much work has been done to develop automatic or other computational methods for their detection, mostly using machine learning, such as Autovetter, Robovetter, and Astronet-Triage. However, these models are typically created and optimized to run on expensive, professional-level hardware, systems that are unavailable to the average person — including students without large funding. Since the early 2020s, as a result of increased interest in AI (artificial intelligence), ML (machine learning), and personal computing power (consider video games or video editing), many people now have the capability to run neural networks (NNs) at home. Consumer offerings from big industry players mean that now almost anyone with a computer can run inference with useful models.

In this paper, we focus on developing a deep learning classifier to help computationally identify exoplanet candidates in data from the TESS mission. We introduce FLICKER (Fast Light-curve Inference and Classification for Keplerian Exoplanet Recognition, a dual machine learning neural network designed around consumer hardware constraints, with customization built in to allow easier adaptation to any device. FLICKER focuses on being primarily a vetting tool as opposed to a final classification tool, where anyone can quickly and easily get predictions to filter down their personal data for manual review. 

Our model adapts the architecture introduced by Tey et al. in 2023, known as Astronet-Triage-v2 \[3\]. Its predecessor, introduced by Yu et al. in 2019, Astronet-Triage has been \[4\] used by the TESS QLP (Quick-look pipeline)\[5\] as a primary filter for exoplanet candidates. Both achieve stellar results, but require too much compute to be feasible on consumer hardware. 

This paper is organized as follows: In Section 2, we describe the input transit signals, as well as how they are labelled and collected; In Section 3, we describe how the data is preprocessed before being fed to the model, with a specific emphasis on the adaptations made to reduce computational complexity required in further steps; In Section 4, we describe the architecture of the neural network, as well as the training process; In Section 5, we quantify the performance of the network, and discuss these results in Section 6; Finally, we conclude in Section 7\.

## 2 Data

For training and testing the model, we use approximately 25,000 human-vetted transit signals across the night sky. To reduce the amount of redone work, this data is taken from the dataset used in the training and testing of Astronet-Triage-v2, which in turn used signals detected by the QLP.  Sections 2.1, 2.2, 2.4, and 2.5, therefore, briefly describe the methods used by prior researchers to obtain their datasets, which is useful for a better understanding of how this model differs.

### 2.1 Converting TESS Images to Threshold Crossing Events

During its Primary Mission (July 2018 to July 2020), TESS \[1\] collected full-frame images (FFIs) of the entire sky every 30 minutes for the entire two-year period. Following this, the FFI cadence was updated to every ten minutes for the first extended mission from July 2020 to September 2022\. QLP produces light curves from these images for all the observed targets in the TESS Input Catalogue (TIC) with an observed magnitude \[6\] (TESS magnitude) brighter than 13.5. Flux time series — essentially the raw brightness at each time, or more specifically, the rate over time at which light energy is received — are computed across 5 different-sized apertures for each star.

These raw light curves are then filtered and massaged (detrended) to remove low-frequency variability originating from various sources, and then combined with curves from similar locations to create a normalized map of light curves around the sky. Each star uses the optimal aperture size based on its TESS magnitude, where fainter stars use smaller aperture sizes to reduce variability introduced by brighter stars.

QLP searches across the light curves using a box least squares (BLS) algorithm to detect potential threshold crossing events (TCEs) \[7\]. The least squares algorithm is a regressive method that determines the line of best fit for a given graph, while the “box” part is used to identify boxy-shaped dips in the curve. For each detected signal, the BLS implementation can heuristically compute additional variables that describe more about the star/exoplanet system. For example, using known values for a specific star’s size, magnitude and distance, we can determine the orbital radius and period for the planet. By analyzing the slope on the edge of a TCE in a similar trapazoidal least squares method, the size of the planet,  transit depth (percent of light blocked), and transit duration can also be found.

Using this information, we can pre-filter most TCEs. First, events that are too noisy are discarded, such that only clear signals are kept for evaluation, as transit-like signals would require a clear dip in brightness. Second, we can filter out instances where the “planet” would orbit within the star’s radius. This handles things like natural variations in a star’s brightness, or where a second, unrelated star happens to line up with the observation point.

### 2.2 Collecting a Set of Labelled Signals

Even with the filters previously described, it would be impractical to want to label every single TCE observed by TESS. The researchers behind Astronet-Triage-v2, therefore, selected \[3\] a subset of TCEs that can be used for training and testing. The dataset was assembled across three batches drawn from the first three years of the TESS mission. Year one contributed 8,992 TCEs from Sector 13 alone — albeit this was a pragmatic choice after hundreds of hours of human labelling had gone into them — which, despite its narrow origin, still managed to contain a diverse range of data artifacts and pointing strategies as the spacecraft itself was settling in and determining the best method for observation. Year 2 added 13,372 TCEs from Sectors 14–26, selected by prioritizing the brightest targets by TESS magnitude, benefiting from the more uniform data characteristics of that observing period. Year 3 added a further 2,588 TCEs from Sectors 27–39, broadening both sky coverage and brightness range.

![Figure 1 \[3\]: Each TESS observation sector is mapped into the sky](https://content.cld.iop.org/journals/1538-3881/165/3/95/revision1/ajacad85f1_hr.jpg)

*Figure 1 \[3\]: Each TESS observation sector is mapped into the sky.*

Observing this figure from the Astronet-Triage-v2 paper, we can see that the data covers a large portion of the night sky. Each black icon on the chart indicates an observed and labelled TCE. The right ascension (RA, x-axis) is measured in hours, specifically the amount of time it would take Earth to rotate from RA \= 0\. The declination (y-axis) is measured in degrees away from the equator of the Earth. Vancouver BC sits at a declination of \~49 degrees, while the RA changes every moment.

Because each batch was selected under different criteria and under different instrumental conditions, the three years contain slightly different data characteristics. For example, TESS reduced its momentum dump frequency from as needed, to every 4.4 days, to once per orbit (13.7 days). However, the authors of the Astronet-Triage-v2 paper note that they do not expect this to negatively impact model performance. Together, the three years of data combine to \~25000 labelled and reviewed datapoints, with a wide range of sky positions and stellar magnitudes.

### 2.3 Labels and their Definitions

For each TCE, the authors of Astronet-Triage-v2 assigned one of five labels: E, B, S, J, or N. For our model, we found that spreading the network slope across 5 outputs was unnecessary, and therefore reduced it to just 4 outputs. The 4 labels used by FLICKER, and in the rest of this paper, are therefore as follows:

1. “E” represents a periodic eclipsing signal. While it is tempting to immediately assume this denotes an exoplanet positive, it may also represent contact eclipsing binary star systems. Contact is where the stars are so close together that they are essentially touching each other, while eclipsing means that the orbital plane of the stars is such that the more luminous of the two periodically occludes the less luminous one.  
2. “B” represents non-contact binary systems. These are distinguishable from E binaries as they generate a more gentle and continuous slope.  
3. “S” represents events containing only a single transit, or where the predicted period is incorrect compared to that determined by the BLS. This might be caused by poor measurements or some other physical object that happens to transit between the star and satellite, like an asteroid, dust cloud, or another satellite (natural or artificial).  
4. “J” denotes junk or not sure. This includes other astronomical events, like stellar variability or instrumental interference from scattered light or cosmic rays. Consider reflections from the Earth, Moon, or Mars, for example. This label is conjoined with “N” from the original set, where no conclusive label decision could be made, or where a weak signal was inconclusive between E and J.

It should be mentioned that Tey et al. relate that the labels are not completely decisive \[3\]. For example, when resolving ambiguous cases between E and S, if there is uncertainty in the period, E was the default label. If there is only one transit outside of the expected transit windows, or if the period potentially includes a secondary eclipse (indicating a combination of more than two planets and/or stars), E is also chosen. 

### 2.4 Data Separation

Tey et al. manually labelled TCEs using visual representations \[3\] over 2 years, with conflicting labels discussed when at least one reviewer chose categories E or S. For targets with only B, J, or N votes, labels were weighted by vote count. This multi-reviewer process was designed to minimize labelling errors and create a high-quality dataset. 

![Figure 2 \[3\]: Example E and B classification charts](https://content.cld.iop.org/journals/1538-3881/165/3/95/revision1/ajacad85f3a_hr.jpg)  

*Figure 2 \[3\]: Example E and B classification charts.*

These charts show examples of signal data along with their determined label. For example, we can observe that E examples are characterized by a relatively long, flat stretch, followed by a deep dip below which quickly recovers to a normal level. In contrast, B examples show large sweeping curves, with no visible “baseline” height. 

Tey et al. also randomly separated the dataset into a training set, evaluation set, and testing set, and recorded this random split to better support repeatability across different training and testing runs.

| Dataset | J | E | B | S | Total |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Training** | 15,912 | 2,102 | 1,681 | 224 | **19,919** |
| **Evaluation** | 2,015 | 216 | 198 | 17 | **2,491** |
| **Testing** | 2,032 | 250 | 200 | 34 | **2,516** |

*Figure 3: Dataset class occurrence breakdown.*

### 2.5 Curve Distribution

Out of the 24,926 labels, most are categorized as junk (J, 19,329). Interestingly, the number of signals identified as eclipsing objects (E, 2613\) is similar to that identified as contact binaries (B, 2079). The figure below is from the work of Tey et al \[3\]., where they maintained a separate N class.

## 3 Preprocessing

For each transit event, we must first preprocess the raw flux data to allow the model to generalize in training and predictions.

### 3.1 Time Series Data

For time series data, the preprocessing step constructs a series of 4 views. These views are each a list of data (array) with some set length that represents some window of the light curve as seen from some particular geometric perspective. For example, imagine trying to analyze an entire book (global view) within a single paragraph, and how your level of detail would compare to looking at just a single chapter (half-phase view), or at just a single page (local view). Furthermore, because the raw flux data varies in length — some stars were recorded for longer than others, with different coverage, cadence, and sector count — the views serve as a standardization step, collapsing each TCE into a more uniform representation that the model can process and generalize. Each view contains some number of bins, which store summary statistics computed from all the flux points that fall within it, rather than individual measurements. This makes the representation more robust to the specific number of observed transits and the exact spacing of data points.

##### 3.1.1 Data Refinement and Normalization

Before any view can be constructed, each light curve must first go through four preprocessing steps: loading and cleaning, detrending, phase folding, and epoch refinement.

Each TCE is associated with a TIC ID, essentially a number that links it to its host star. For a given star, there may be one or more FITS (Flexable Image Transport System, essentially TCE data), each corresponding to a different TESS observing sector, so we have to first collect all available files. We then use Lightkurve, an open-source FITS tool \[8\], to load the files so they can be processed. Doing so, we discard any data points where the flux is not a usable number, and concatenate the remaining time and flux arrays across sectors into a single, time-sorted sequence. For completeness, for stars observed across multiple sectors, all available data contribute to the same light curve. 

Raw flux time series contain long-term brightness trends caused by stellar variability, spacecraft specifics, instrumental effects, or other factors. These low-frequency signals are irrelevant to transit detection, and if left in, could obscure the shallow dips that the model has to learn to identify. To remove them, we apply Wotan’s biweight location estimator \[9\] with a sliding time window of 0.5 days to produce a flattened flux series where the slow trends have been divided out. Importantly, before calling the detrending function, we compute a mask that marks the expected transit windows using the BLS-reported period, epoch, and duration. The mask is used to “hide” sections in the data that contain the dips that we are looking for, such that the flattening step does not adjust or remove the transit signal; Only the unmasked, out-of-transit points are used to fit the detrending model.

Once detrended, the flux series is phase-folded. By convention, we fold based on the sequence of timestamps within a TCE  and maps each one to its position within a single orbital period, expressed as a fraction offset between \-0.5 and 0.5 where 0 is the moment of the centre of the observed TCE. Points from many separate transits, spread across weeks or months of observation, are therefore collapsed into a single orbit representative of the average behaviour of the star system observed, much like a patterned fabric being stacked together to overlap repeating features.

Because the BLS-estimated epoch is a statistical estimate rather than a precise measurement of the transit centre, we then perform an epoch refinement step to more accurately fold around the correct points in the data. This is done by sliding a window of width equal to the transit duration along the phase fold flux, and the position where the window average is lowest is identified as the true transit minimum. The phase array is shifted so that this minimum always lands at phase \= 0\. Without this, small period errors might build up and cause the edges of the transit dip to appear slightly offset in every view, therefore blurring its edges after binning and across orbits.

##### 3.1.2 View Construction

With the cleaned, folded, and aligned flux computed, we now construct four views for each TCE. In contrast to many larger models, we use a single detrending pass and produce a minimal set of 4 views as part of our efforts to produce a less computationally expensive model; Astronet-Triage-v2 uses 3 specialized detrending passes and 21 independent views \[3\], for example. This consequently requires a higher level of usefulness that each view must provide, so we select the 4 most variable and data-rich views to provide the model with enough data. Furthermore, just four views keep preprocessing more tractable, and significantly reduce the model’s input dimensionality. However, this is a tradeoff, where doing so makes it more difficult for the model to confidently train and discern differences.

The global view represents the full orbit, binned into 201 equally spaced bins from within the phase of (-0.5, 0.5\]. For each bin, we record the median and standard deviation of all flux points falling within it. We also record two binary flags per bin. The first indicates whether the bin lies inside the transit window, defined as within 1.5 transit durations of phase \= 0, and one indicating whether the bin contains any data at all, since sparse coverage can leave some bins empty of recordings. Empty bins are assigned a median value of 1, and a standard deviation of 0 to represent a flat, unremarkable and trendless baseline. To normalize the global view, we subtract the median flux of the out-of-transit bins (very nearly 1), then establish the quiescent stellar brightness at a baseline of 0, and then divide so that the minimum value in the view is one. This normalization step is important to generalization, as it tells the model to train and look for a consistent pattern observed in the brightness, rather than the raw brightness values that depend on each star. The global view has an output vector metric of 201x4 \= 804 values.

The local view uses the same binning approach, but zooms into a window width of \+/-2 transit durations on phase 0, using 61 bins. This provides a higher resolution look at the shape of the transit — consider the ingress, flat bottom, and egress — which the global view cannot resolve as clearly. It is then normalized in the same way as the global view and is saved with an output vector matrix of 61x2 \= 122 values.

The secondary view is designed to capture a significant brightness dip occurring outside the primary transit window. For an eclipsing binary star system, a secondary eclipse caused by the dimmer star passing behind the brighter one produces a second dip at a phase determined by the binary’s orbital shape. To find this dip, we exclude the \+/-2 transit duration used for the local view, and then slide a window with transit duration width across the remaining points. The window computes the mean flux at each position, and identifies the lowest mean as the secondary eclipse candidate. A 62-bin window is then centred on that phase, and constructed and normalized in the same manner as the local view, to an output vector dimension of 61x2. Planet transits do not produce a secondary eclipse, so the secondary view is expected to be flat for planet candidates and structured for binary systems.

The half-period view folds the flux at half the reported period, rather than the full period, then bins a \+/-2 transit duration window around the resulting phase \= 0\. Geometrically, an eclipsing binary with a nearly symmetric orbit has its primary and secondary eclipses separated by roughly half a period, as it takes the same amount of time to perform the first half of the orbit as the second half of the orbit. Knowing this, we fold the data at half the period, such that both eclipses map onto the same phase. Here, binary systems reinforce each other’s data and generate a deeper (or doubled for lopsided binaries) feature. Conversely, for a planet transit, only the transit itself maps to phase zero, leaving the rest of the view flat. Only the standard deviation is recorded per bin in this view, and the median would become unreliable when folding a transit signal over non-transit data, and provide a fuzzy halfway marker rather than useful information. Again, this view is normalized in the same manner as the local view, and given an output vector matrix of 61x2 \= 122 values.

Finally, we produce a set of two half views that provide the network with a comparison between “odd” and “even” events. This view splits transit events into their orbit index, splitting the phase-folded data into alternating orbits around the reference epoch. Both halves are binned over the \+/-2 transit duration window as the local view, using the same 61 bin count. However, rather than a single median and standard deviation pair, four channels are recorded per bin, with a pair for both odd and even orbits to give an output vector matrix of 61x4 \= 244 values. Each median channel is normalized in the same manner as the local view, and the corresponding standard deviation is scaled by the same factor. Physically, a real exoplanet produces the same dip every orbit, so odd and even profiles should look nearly identical. An eclipsing binary, however, alternates between a primary eclipse and a secondary one — the deeper and shallower events land on different orbit parities when folded at the same period, making a visible depth difference between even and odd orbits. This view is also excluded by the time-flip augmentation, since odd and even are defined by absolute orbit count from epoch, and therefore would carry no physical symmetry under a time reversal.

As a final precautionary measure, all 5 views are clamped to the range \[-5.0, 5.0\] after being normalized. This is necessary because very shallow transits — like those from small planets, or binaries with very similar luminosities — can produce normalization scale factors very close to 0\. During development, dividing by these near-zero numbers made some bins’ standard deviation values on the order of $10^{29}$, understandably overwhelming the convolutional towers of the network.

## 3.2 Scalar Data  
In addition to the four time-series views, each TCE is accompanied by a vector of scalar features drawn from the BLS output and the TIC. These include the orbital period in (Earth) days, transit duration in days, and other data that might be useful for the network. 

Scalar features are z-score normalized using the mean and standard deviation computed across the training split only, ensuring that no information from the evaluation or test sets influences the normalization in training. Not all stars in the TIC have complete entries; stellar radius is missing for a large number of TCEs in the dataset, for example. To resolve this, missing values are imputed with the training dataset mean for that feature before normalization, so planted entries land at a z-score of 0 rather than requiring special handling by the model.

The model receives 12 types of scalars that describe pieces of data derived from each TCE, for a final vector matrix of 12x1. Specifically, the scalars are arranged as follows: 4 describe the transit event itself as reported by the BLS, being the orbital period in days, transit duration in days, transit depth in parts per million, and the number of complete orbital periods observed across the full light curve — a longer period provides context about how often and how repeatedly the transit occurs, and therefore how many transit events contributed to the folded signal, while depth and duration constrain the physical size and geometry of the occluding body. 3 describe the host star, including its TESS magnitude (apparent magnitude of the star at the recorded wavelengths of light on TESS), mass in solar masses (x times the mass of the Sun), and radius in solar radii (x times the radius of the Sun), all drawn from the TIC — stellar radius is particularly useful as it can help the model discern a small eclipsing companion since the transit dip is expressed in terms of the star’s own radius. The final 5 scalars are derived during preprocessing, rather than being read from external databases. The normalization scale factors from the global, local, and secondary views are each included individually, because the views themselves are optimized for pattern matching through normalization; the scale factor is the only record of how deep the original transit was in actual physical units. The phase of the identified secondary ellipse candidate is also included to provide the model with the location of the secondary view. Finally, the number of data points collected is passed so the model can assess the credibility of the data in its confidence and confusion analyses. 

### 3.3 Data Augmentation

While almost 25,000 data points is a large amount for a human — or team of humans — to evaluate and train on, it is a relatively small number of examples for a machine learning model to learn from. On “small” datasets, it is often easier for models to “memorize” the features of a dataset instead of learning how to generalize patterns to new test cases. Very simply, a neural network can quickly find a relative maximum performance for its specific input, then spend the rest of its training further memorizing that input. When the model is later exposed to new data, it cannot perform well since the new data does not exactly match the old.

One common method to deal with this problem is to artificially augment the data that is provided to a model. Augmentation is not the ultimate solution; Types of augmentation are relatively simple and constrained, but these can still encourage a network to look for patterns rather than specific details. For example, in a computer vision model learning to separate cats and dogs, you can augment your training data by horizontally flipping images or adding small amounts of random noise to the colour data of images. 

For our model, we can apply two augmentations fairly easily at training time. First, we execute a random time flip on some of the input views provided to FLICKER’s trainer. Specifically, each of the global, local, and half-period views is given a ½ chance to flip, where they are provided to the model in the reverse direction. Since the planet’s transit curve is symmetrical — or at least is so before noise and interference are added — we can safely reflect them along the time axis. The secondary view is explicitly not flipped for the same reason. Because it is centred on the secondary phase, rather than phase 0, such a reflection would be geometrically inconsistent. Second, we add an amount of Gaussian noise to all the view channels and to the scalar vector. We use Gaussian noise specifically as it provides a noise function that more closely approximates real-world variance, such that the values the noise takes across some given area match a Gaussian bell curve, rather than a random distribution. The strength of the noise is also scaled to the distribution of points within the view or scalar, where larger distributions receive stronger noise to approach a similar ratio of variance.

### 3.4 Data Example

Allow us to recall the example charts given by the paper in Section 2\. We can generate similar graphs using the precomputed data that is to be given to the model. Note that while the last view is displayed with normalized flux, the model does not get this information, only the standard deviation, as shown with the light blue background. For example:  

![Figure 4: Data chart for TIC 90104045 (E)](https://pierceoxley.ca/flicker/image-references/90104045.png) 

*Figure 4: Data chart for TIC 90104045 (E).*

Observing, we notice a consistent, flat light curve except for the transit events. We can see that the local and secondary views are very similar, and that they align in the half-period view. Furthermore, we see that the even and odd views line up quite well, indicating that this cannot be a binary system. 

## 4 Network Design

Our model uses a convolutional neural network architecture derived from Astronet-Triage-v2 \[3\]. The model is comprised of two stages: First, a set of parallel convolutional towers processes each view independently. Second, a fully connected network combines its outputs with the scalars to provide a final classification. 

![Figure 5: High level network design](https://pierceoxley.ca/flicker/image-references/network_design.png)  

*Figure 5: High-level network design.*

### 4.1 Architecture

Each of the 5 views is processed by its own convolutional tower, where each tower consists of three blocks. Each block applies a 1D convolution with a kernel width of 5, followed by batch normalization, a ReLU activation, and a max pooling factor of 2\. The three blocks use 16, 32, and 64 filters, respectively, so each successive block captures increasingly abstract patterns over a wider range of the input. To match the data input as described in Section 3, the global view and even/odd towers fill 4 input channels, while the local, secondary, and half-phase views each take 2 channels. The output of each tower is flattened into a 1D vector, and all 6 are concatenated together with the 12 scalar features into a single combined representation.  

![Figure 6: Convolutional tower design](https://pierceoxley.ca/flicker/image-references/network_tower.png) 

*Figure 6: Convolutional tower design.*

This combined vector is passed through a fully connected network (FC) consisting of three hidden layers, trimming down from 2380, to 256, to 128, to 64\. Last, the final layer maps the hidden representation to 4 output logits, 1 for each of the classes identified in Section 2\. Each hidden layer is followed by its own ReLU activation and dropout layer of 0.5, where said dropout randomly sets 0.5 of the output width to 0\. This is done to discourage the network from relying too heavily on one or two single features, and helps reduce overfitting. The dropout layers are disabled during evaluation and inference, so predictions are deterministic as opposed to depending on a random dropout chance.  

![Figure 7: Fully connected network design](https://pierceoxley.ca/flicker/image-references/network_fc_block.png)  

*Figure 7: Fully connected network design.*

### 4.2 Training

The model’s training parameters are made easily accessible at runtime through the use of Python arguments to better support the goal of an easy-to-train and run model on any device. With that said, any values mentioned within this section, and any model results discussed in this paper will use the experimentally determined bests — the defaults as provided in the final software.

The model is trained for $2 \times 10^5$ gradient steps using the Adam optimizer \[10\] with a scheduled learning rate and weight decay of $10^{-3}$. The learning rate is based on PyTorch's cosine annealing \[11\], set to decay from $10^{-3}$ to $10^{-6}$ over the full step count, while the weight decay is simply to add a small penalty for excessively large weights. As the training set is so heavily imbalanced, we use a cross-entropy loss function, where each class is assigned a weight inversely proportional to its frequency. This is softened by an exponent of 0.75 to avoid over-correcting for the rarest cases. Such weighting means that misclassifying an E signal or S  signal contributes proportionally more to the loss than a misclassification of a J signal, guiding the model to pay more attention to the minority cases. There is a further 2x weight applied for E labels to encourage the model to guess E over other options when it is unsure. This comes at the expense of precision, but improves recall. Gradients are clipped to a maximum normal of 1 at each step to prevent unstable updates early in training when the loss surface is poorly conditioned and understood.

The model checkpoint saved for each training run is determined by an evaluation set every 500 steps, and the checkpoint with the highest E-class area under the precision-recall curve (AUC-PR) is retained on disk. This ensures that the saved model is the one that best recovered eclipsing candidates, rather than the one that simply completed the most steps (and may or may not have begun to overfit).

### 4.3 FLICKER Solo

FLICKER Solo consists of a single trained model. It is intended for cases where speed or hardware constraints make running or training multiple models impractical. Producing a prediction for a new TCE only requires one forward pass and loading a single model into memory. 

Solo is the best choice for when the user’s computer does not have a dedicated GPU or neural accelerator, or lacks sufficient memory to run Choir. Consider a Raspberry PI, or a mobile phone.

### 4.4 FLICKER Choir

FLICKER Choir consists of an ensemble of 10 models, each trained from a different random seed. Because the training process involves (pseudo) random weight initialization and batch ordering, two models trained on the same dataset will converge on different solutions, making different kinds of errors on different examples; Aggregating their predictions reduces the influence of any individual model’s random mistakes. At inference time, each model produces a set of class probabilities via a softmax over its logit outputs, and these are combined to generate a single ensemble prediction. Further, Choir has two separate methods for determining this final prediction: Choir Baritone and Choir Soprano. Choir Baritone optimizes for consistent model accuracy, and therefore averages together all 10 probability vectors into one ensemble prediction. Choir Soprano takes each model’s results individually, and assigns E if any one of the 10 assigns E.

Choir produces more reliable results than Solo, particularly on ambiguous or borderline examples, though it does so at 10 times the training, inference, and memory cost. With this said, Choir is still more than capable of running on entry-level machines, so long as they have sufficient capacity to hold the models in memory, and they have a processor optimized for parallel compute to run it on. Consider a gaming computer or laptop with a dedicated GPU.

## 5 Performance

Here, we report the performance and other results of our model, discussing the metrics themselves and what they represent. 

To best evaluate the different models, we determine the confidence threshold at which the network’s F2 score is maximized; F2 is a type of F𝛽 metric, scored from 0 up to 1, in which missing a positive case is weighed more heavily than accidentally flagging a negative case as positive. Specifically, positive cases are weighed 2 times more than negative ones. This type of score prioritizes high recall, as further explored in Section 5.1. We optimize for F2 performance based on a confidence threshold of 0.29.

To summarize our results, we produce the following table denoting the variations between models and their performance. Metrics are given between 0 and 1:

| Model | Precision | Recall | Miss Rate | AUC-PR |
| :---- | :---- | :---- | :---- | :---- |
| **Solo** | 0.474 | 0.9139 | 0.0861 | 0.8777 |
| **Choir Baritone** | 0.518 | 0.9221 | 0.0779 | 0.895 |
| **Choir Soprano** | 0.372 | 0.9713 | 0.0287 | 0.890 |

*Figure 7: Model performance summary metrics.*

### 5.1 Compute

FLICKER was trained using a consumer PC with freely and widely available software, all running on Windows 11 Pro for better remote access. We expect similar performance on any modern operating system with proper hardware support. The hardware used for this was an Intel i7-14700KF CPU, a single ZOTAC GAMING GeForce RTX 3060 Ti Twin Edge OC (8 GB VRAM) GPU, 48 GB of DDR4 RAM, and an ASRock B660M Pro RS Motherboard, along with NVMe and HDD storage as needed — the full dataset from Zenodo is \~25 GB, plus \~225 MB for the preprocessed data.

#### 5.1.1 Preprocessing

When preprocessing the data, performance depends heavily on the number of concurrent workers assigned. With 16 workers, the CPU manages the full dataset in $1.329 \times 10^3$ s or 22.15 min, at a total rate of 0.86 worker seconds per TCE. System memory is found to be around 1.1 GB worker$^{-1}$ (1.1 GB per worker). We expect linear performance with respect to worker count, so long as system memory and CPU performance are not bottlenecked.

#### 5.1.2 Training

When training the model on GPU with CUDA, we observe the following metrics. Throughput averages around 76 steps s$^{-1}$ or 13.2 ms step$^{-1}$. For a complete training run of FLICKER Solo across $2 \times 10^5$ steps, this totals 363 s, or \~4.4 min. For the 10 model Choir run with $2 \times 10^6$ steps, this comes to 2631 s, or \~44 min. These metrics assume no performance degradation due to thermal losses. VRAM peaks at 46 MB while training, meaning compute is the bottleneck rather than memory usage. System memory usage is nearly constant at 7558 MB for each model.

#### 5.1.3 Inference

To evaluate the inference performance of the various FLICKER models, we ran the models across the test dataset of 2431 TCEs and observed the following metrics:

| Model | Device | Compute Time | Throughput | Per TCE |
| :---- | :---- | :---- | :---- | :---- |
| **Solo** | GPU | 418 ms | 5,818 TCE s$^{-1}$ | 0.17 ms |
| **Solo** | CPU | 2,709 ms | 897 TCE s$^{-1}$ | 1.1 ms |
| **Choir** | GPU | 3,362 ms | 7,231 TCE model s$^{-1}$ <br>(7,231 TCEs per second per model) | 0.14 ms model$^{-1}$ |

*Figure 8: Model inference summary metrics.*

When performing inference, we observe a peak VRAM usage of 29 MB. System RAM usage peaks at 1.4 GB model$^{-1}$, and averages around 900 MB model$^{-1}$.

### 5.2 Precision, Recall, and E AUC-PRw

In machine learning, precision and recall are used to evaluate a model's accuracy against the number of correct answers. Both are scored out of 1; perfect precision means that every guess a model makes is correct, while perfect recall means that the model correctly identifies every desired class. For a model with four outputs, we would expect random guesses to provide precision and recall scores of 0.25 each (4$^{-1}$). To compare these two values, a curve is constructed with corresponding precision and recall values, where a perfect model achieves an AUC (area under the curve) of 1.0. With four outputs, random guessing would yield an AUC-PR of 0.625 (0.25 x 0.25). Precision and recall are competing factors, such that a higher value of one tends to have a lower value of the other associated with it. For use in FLICKER, we adapt this AUC-PR to evaluate specifically for the E class — while our models can provide a more standard AUC-PR, we are only interested in classifying signals as E or not E. Recall that E TCEs are most often composed of exoplanets and eclipsing binaries. 

#### 5.2.1 Solo

On the test set at an optimized threshold of 0.29, we achieve an average AUC-PR of 0.8777 \+/-0.009. Observing, we see that our precision stays high for recall rates under 0.56, then linearly tapers off, reaching equivalence at 0.80.

![Figure 9: Solo E AUC-PR Curve (0.8777)](https://pierceoxley.ca/flicker/image-references/pr_curve_solo.png)

*Figure 9: Solo E AUC-PR Curve (0.8777).*

5.2.2 Choir Baritone

On the test set, we achieve an optimized E AUC-PR of 0.8914 \+/-0.004. We see that Baritone achieves a higher E AUC-PR, likely due to its higher confidence associated with multiple models working together. Shapewise, it is very similar to Solo, just slightly more pronounced towards 1.0, 1.0.

![Figure 10: Baritone E AUC-PR Curve (0.8914)](https://pierceoxley.ca/flicker/image-references/pr_curve_baritone.png)

*Figure 10: Baritone E AUC-PR Curve (0.8914).*

#### 5.2.3 Choir Soprano

On the test set, we achieve an optimized E AUC-PR of 0.8918 \+/-0.005. Here we observe the most visible curve in the graph, meaning that precision stays relatively high until starting to steeply drop off at higher recall than the other models.

![Figure 11: Soprano E AUC-PR Curve (0.8918)](https://pierceoxley.ca/flicker/image-references/pr_curve_solo.png)

*Figure 11: Soprano E AUC-PR Curve (0.8918).*

#### 5.2.4 Threshold

When we graph miss rate as a function of the models’ respective confidence thresholds, we compute the following graph:

![Figure 12: Miss rate threshold sweep for each model](https://pierceoxley.ca/flicker/image-references/miss_rate_vs_threshold.png)  

*Figure 12: Miss rate threshold sweep for each model.*

In this graph, the miss rate of each model is computed at confidence intervals of $5 \times 10^{-4}$ and interpolated horizontally across until the next point.

We visually note three thresholds: 0.0105, 0.215, and 0.029, where 0.0105 and 0.215 were given in Astronet-Triage-v2 paper, and 0.29 was found to be ideal by optimizing the F2 result. Further, we note that Soprano achieves a recall of 100% (miss rate of 0%) at a confidence level of 0.0105, identical to Astronet-Triage-v2. At a confidence of 0.215, Soprano achieves a recall of 97.95%, exceeding that of Astronet-Triage-v2 at 96.9%. However, we find our recall at the large expense of precision. At the two benchmarks, we get 11.25% and 27.28% precision, while Astronet-Triage-v2 gets 41% and 79.8%, respectively.

### 5.3 Confusion Matrices

Confusion matrices are used to evaluate a network's performance across the different classes of its output. Specifically, we can use them to identify areas where the model performs well or struggles to identify the correct class. In training, we look at areas of high confusion and attempt to design our network to combat these areas. A perfect model generates a confusion matrix with 100% understanding along the diagonal from the top left to the bottom right, as it correctly identifies every class, every time.

##### 5.3.1 Solo

Across the test set, we produce the following matrix with FLICKER Solo:  

![Figure 13: Solo confusion matrix (percentage)](https://pierceoxley.ca/flicker/image-references/confusion_solo_pct.png)

*Figure 13: Solo confusion matrix (percentage).*

Observing, we notice that the model achieves high confidence for E, B, and J classes, with low confidence in S classifications. We can also see that the network has a strong tendency to guess E when it is unsure. This is expected with the way that training is weighted, and how we prefer over-guessing E to improve recall.

#### 5.3.2 Choir Baritone

Across the test set, we produce the following matrix with Choir Baritone:

![Figure 14: Baritone confusion matrix (percentage)](https://pierceoxley.ca/flicker/image-references/confusion_baritone_pct.png)

*Figure 14: Baritone confusion matrix (percentage).*

Similar to Solo, we notice that the model achieves high confidence for E, B, and J classes, with low confidence in S classifications. Important differences include a \+0.8% confidence in E, \-6.1% confidence in S, and a \+9.1% confidence in S cases misclassified as E. Since 10 models are deciding in a group consensus, each model might have higher confidence in a particular “strand” of S examples, which combine to produce a higher net confidence. We also see a \+9.1% increase in prediction rate for S. This is expected as a result of heavy favouring towards E classes, though, favourably, single transit events are much easier for human evaluators to pick out from E than B, for example.

#### 5.3.3 Choir Soprano

Across the test set, we produce the following matrix with Choir Soprano:

![Figure 15: Soprano confusion matrix (percentage)](https://pierceoxley.ca/flicker/image-references/confusion_soprano_pct.png)

*Figure 15: Soprano confusion matrix (percentage).*

Similar to Baritone, we notice again that the model achieves high confidence for E classes, though conversely looses confidence in B and J, while gaining hugely in S. Specifically, \+5.7% in E, \-15.2% in S, \-5.7% in B, and \-9.1% in J when respectively compared to confusion in Solo. These results show the somewhat unpredictability of Soprano; Similar to Baritone, the model is highly trained to favour E, and does so at the cost of even more confidence loss in the S class. The model has effectively learned that any sharp dip in a given light curve is likely an exoplanet — likely due to data imbalance and training setup — and therefore will guess E even when only one is observed.

## 6 Discussion

The results presented in Section 5 describe the influence of two competing forces on FLICKER’s performance — the model is very good at finding eclipsing signals, though it is quite uncertain in its predictions.

### 6.1 Precision Gaps

The most significant difference between FLICKER and larger industry models is precision. At the two reference thresholds of 0.0105 and 0.215, we recall that Choir Soprano achieves recall of 100% and 97.95%, respectively, matching or exceeding Astronet-Triage-v2's 100% and 96.9%. The corresponding precision values, however, tell a different story: at those same thresholds, Soprano achieves 11.25% and 27.28%, while Astronet-Triage-v2 achieves 41% \[3\] and 79.8%. This gap is resolved in two ways, both as an intentional choice and a structural limitation of the model design.

FLICKER’s training applies an additional 2x weight to E labels on top of the class-frequency corrections to drastically push the model to guess E when it is uncertain. The operating threshold is also chosen to maximize F2 score — a metric that weighs recall twice as much as precision. We expect that it is far more useful to most researchers to receive a comparatively lengthier yet highly inclusive list as opposed to a comparatively shorter yet more lossy subset of TCEs.

The second cause of the precision gap is a structural imitation. Astronet-Triage-v2, for example, uses 21 views produced from three independent detrending passes; FLICKER uses 5 across only 1 detrend operation. With more data available to the model, the larger designs tend to have larger working signals. Consider, for example, a TCE where the global and local views are ambiguous. FLICKER would likely be unable to discern a meaning from this example and would default to the safe E value. FLICKER has to make the same decisions from a much smaller amount of data, where this lack of confidence is shown most clearly as decreased precision between similar TCEs, like when the model cannot tell a shallow planet or E transit from a noise artifact that happens to look like one.

#### 6.1.1 Addressing the S Class

Across all three model variants, the S class are the most frequently misclassified, with between 48.6% and 75.8% of them predicted as E. Recall that an S-labelled TCE contains only a single transit event, or multiple where the BLS-estimated period is believed to be incorrect; a single clean dip in a given light curve can easily replicate the shape of a planet if conditions are correct. Further, with fewer and potentially unmatching events in the data, the half-period view provides no reinforcing signal, while the odd/even view and secondary view — which require at least two transits to populate— are similarly unhelpful. The model is therefore left to classify an S example from its local and global shapes alone, which can look identical to an E example. We can teach the model that if the other views are unhelpful, it should go with an S example, but these views are also unhelpful during J examples, or during E examples, where some transits are missed for whatever reason. Prior work has addressed this by computing a separate set of helper views, each containing a unique single transit. In such a case, the model must work across many more views, but it gains more information about the repeatability of a TCE.

One limitation of this method of exoplanet detection is that planets which orbit far from their host star are most often discarded as S examples, due to the length of time that their orbit takes to complete. If we consider that TESS has only been observing the sky since 2018,  planets with orbital periods of less than 4 years would not have experienced 2 transits. 4 years is not a very long time for a planetary orbit, even within our solar system where the outer gas giants have periods between 11.8 and 164.8 years. Furthermore, the gas giant planets, which are by far the largest and therefore the most detectable by transit, tend to form beyond the snow line, far from their host star \[12\]. This means that some portion of S classes may in fact be exoplanets, and could warrant extra investigation with alternate methods of exoplanet searching.

### 6.2 Solo, Baritone, and Soprano

The three model variants essentially represent three different mindsets behind the idea of how much agreement is necessary for the model to have before it commits to E. Observing from the test results in Section 5, and as a result of its design, Solo makes conservative predictions since a single mode must be sufficiently confident by itself to cross the threshold. Baritone moderates the group’s collective probability estimates, producing results that are more consistent than any single model while remaining somewhat constrained at low thresholds. Soprano takes the most permissive approach, where any one of the ten models finding a strong signal is enough to flag as E. What the ensemble models gain is most clearly robustness on ambiguous examples, rather than dramatic raw performance gains. Baritone’s AUC-PR improvement is a modest \+0.018, but the \+9.1% improvement in S-class recall suggests that different models in the ensemble develop sensitivity to slightly different variations of the same signal, and their combination produces more confident predictions on examples where any single model alone would be less certain.

Practically, the choice between the models depends entirely on what the end user intends to do with the output. If downstream human review is fast, and a high false-positive rate is acceptable, Soprano is the right tool with its 97.3% recall at the optimized threshold. If the user needs results that are more reliable on their own, Baritone’s higher precision makes it more appropriate. Finally, Solo is the right option when hardware constraints make running multiple models impractical, or when prediction speed is the primary concern.

### 6.3 Computational Cost and Related Consequences

FLICKER’s central goal is to run usefully on consumer hardware; the design choices made in service of that goal have made a considerable impact on the models. Reducing to just 5 views and 1 detrending pass consequently lowers the model’s input dimensionality, cutting processing time and the complexity required in the convolutional towers. These reductions make FLICKER practical on machines that would otherwise be unable to train or run large models in a reasonable amount of time. 

The hardware results from Section 5.1 make a compelling computational argument for these ideas. Training a full Solo model takes less than 5 minutes on a mid-range consumer GPU, and a complete ten-model Choir run finishes in less than an hour. More importantly, the video memory footprint of the model is incredibly small, peaking under 50 MB under training and 30 MB during inference, meaning the model occupies nearly no VRAM when being used. In principle, FLICKER could train and run on hardware with as little as a few hundred megabytes of dedicated GPU memory, within the range of integrated graphics or entry-level discrete GPUs. System RAM is more impacted, though still at a very reasonable level for 1 model, with 10 model inference runs better suited for mid to high-end consumer systems. However, users whose hardware cannot accommodate this can fall back to solo with only a modest performance cost, try running fewer concurrent models, or run the models in series rather than in parallel.

On throughput, inference is similarly accessible. Solo classified nearly $6 \times 10^3$ TCE s$^{-1}$ on GPU during testing, meaning the full 25000 example dataset is processed in under 5 seconds. Even on the CPU alone, Solo can process up to $9 \times 10^2$ TCE s$^{-1}$ on a decent CPU, meaning that a researcher with no dedicated GPU could still classify a personal dataset of several thousand candidates in a matter of seconds. For the use case FLICKER is designed around, where the goal is to reduce a large set of candidates to a manageable shortlist for human review, these speeds mean the models add no meaningful delay to a given research window.

### 6.4 Comparison to Past Works

FLICKER sits between Astronet-Triage \[4\] and Astronet-Triage-v2 \[3\] in terms of architectural complexity and resulting performance. Astronet-Triage used two views and a binary planet or non-planet classification, while Astronet-Triage-v2 expanded to 21 views across three detrends and 5 output labels. FLICKER comes in between, with 5 views, 1 detrend, and 4 outputs. This design improves on the training performance of both noted models, while adapting to better match the compute power expected in modern-day systems; it still recovers most of the signal that the larger models benefit from, but does so with improved efficiency. The only gap is in precision, where the reduced view count leaves the model with less to decide on in similar examples, and where our training decisions push it further toward recall at the expense of said precision.

Direct comparison with Autovetter and Robovetter \[3\] is more difficult, as both tools were designed for Kepler data rather than TESS FFIs; The TESS false-positive rate is higher, meaning performance figures from these Kepler tools do not accurately reflect comparable conformance \[3\]. For the same reason, FLICKER’s precision, while lower than Astronet-Triage-v2, is not an unusual result for a model operating on FFI-derived TCEs, where a high proportion of threshold crossings are expected to be junk by design of how the QLP selects its targets \[1\], \[3\].

### 6.5 Notes for Future Work

The most natural extension of this work, as shown by the results of model benchmarks, is to improve on both S classification and raw precision in order to allow the model to be more confidently used to classify all types of TCE. A set of per-transit views, in addition to or as a replacement for the odd/even view (since this one can be considered to have backup in the half-phase view). Each would contain a single isolated event, along with a timestamp. This approach would give the model more direct evidence about whether the transit repeats consistently, rather than asking it to infer that from an averaged fold or the other views. 

Beyond the model architecture, the training dataset itself is an area for improvement. The labelled set used for this model covers TESS sectors 1-39 \[3\], and TESS has continued observing since, with additional sectors now available through MAST \[6\]. Expanding the training data to cover more recent and updated sectors would broaden the sky coverage and brightness distribution that the model can observe. Consider, for example, that year 3 data already showed slightly different characteristics than the first two years of TESS observations.

## 7 Conclusion

We have presented FLICKER, a dual convolutional neural network designed to quickly and computationally inexpensively identify transiting exoplanet signals from TESS data. As a whole, FLICKER takes preprocessed flux time series data and uses deep learning to assign 1 of 4 output classes. Specifically, “E” for eclipsing signals, “B” for contact binaries, “S” for single transits, and “J” for inconclusive or noisy signals. The model was trained using a set of \~25000 signals, as collected, manually reviewed, and used in Astronet-Triage-v2 \[3\].

FLICKER is designed to be computationally inexpensive, meaning that the model can be trained and run on almost any device. We provide two sub-models, FLICKER Choir and FLICKER Solo, that are tuned for this purpose; Performance or speed can be prioritized for Choir and Solo, respectively. Further, Choir Baritone optimizes for model accuracy, while Choir Soprano casts a wider net to reduce false negatives. Ultimately, FLICKER is designed to help the average person filter down the number of potential light curves they have to analyze.

The three models, Solo, Baritone, and Soprano achieve recall rates of 91.4%, 92.2%, and 97.1%, respectively. They do this at relatively low precision — 0.47, 0.52, and 0.37 respectively — though at exceedingly low hardware requirements. The models can run inference on nearly any modern device, and train in similarly hardware constrained environments if needed.

## References

\[1\] NASA TESS Science Support Center, "What is TESS?," *HEASARC*. \[Online\]. Available: https://heasarc.gsfc.nasa.gov/docs/tess/what-is-tess.html.

\[2\] MIT TESS Team, "Mission Overview," *TESS \- MIT*. \[Online\]. Available: https://tess.mit.edu/science/.

\[3\] E. Tey *et al.*, "Identifying Exoplanets with Deep Learning. V. Improved Light-curve Classification for TESS Full-frame Image Observations," \*The Astronomical Journal\*, vol. 165, no. 3, art. 95, Feb. 2023\. doi: 10.3847/1538-3881/acad85.

\[4\] L. Yu *et al.*, "Identifying Exoplanets with Deep Learning III: Automated Triage and Vetting of TESS Candidates," *The Astronomical Journal*, vol. 158, no. 1, art. 25, 2019\. doi: 10.3847/1538-3881/ab21d6.

\[5\] C. X. Huang et al., "Photometry of 10 million stars from the first two years of TESS full frame images: part I," Research Notes of the AAS, vol. 4, no. 9, p. 204, 2020\.

\[6\] NASA TESS Science Support Center, "TESS Data Products," *HEASARC*. \[Online\]. Available: https://heasarc.gsfc.nasa.gov/docs/tess/data-products.html.

\[7\] G. Kovács, S. Zucker, and T. Mazeh, "A box-fitting algorithm in the search for periodic transits," Astronomy & Astrophysics, vol. 391, pp. 369-377, 2002\. doi:10.1051/0004-6361:20020802

\[8\] Lightkurve Collaboration, "Lightkurve: A friendly package for Kepler & TESS time series analysis in Python," GitHub repository. \[Online\]. Available: https://github.com/lightkurve/lightkurve.

\[9\] M. Hippke, T. J. David, G. D. Mulders, and R. Heller, "Wotan: Comprehensive Time-series Detrending in Python," *The Astronomical Journal*, vol. 158, no. 4, art. 143, Sep. 2019\. doi: 10.3847/1538-3881/ab3984.

\[10\] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in Proc. 3rd Int. Conf. Learning Representations (ICLR), San Diego, CA, 2015\. arXiv:1412.6980

\[11\] A. Paszke *et al.*, "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2019\. doi: 10.48550/arXiv.1912.01703.

\[12\] G. M. Kennedy and S. J. Kenyon, "Planet Formation around Stars of Various Masses: The Snow Line and the Frequency of Giant Planets," *The Astrophysical Journal*, vol. 673, no. 1, p. 502, Jan. 2008, doi: 10.1086/524130.

## Appendix

The complete repository, including code, writing, benchmarks, and more, as developed for this project, can be found on GitHub (https://github.com/Parcley27/FLICKER) or otherwise pulled from [git.pierceoxley.ca/FLICKER](git.pierceoxley.ca/FLICKER). Images generated for this paper are available at [https://github.com/Parcley27/FLICKER/paper/image-references/](https://github.com/Parcley27/FLICKER/paper/image-references/), and pulled from [pierceoxley.ca/flicker/image-references/](pierceoxley.ca/flicker/image-references/) for use in markdown documents.

A collection of acronyms and definitions that may be useful to recall for this paper is found as follows:

* Adam: Adaptive Moment Estimation  
* AI: Artificial intelligence  
* AUC-PR: Area Under the Curve of Precision-Recall  
* BLS: Box Least Squares  
* Detrending: Removing slow, long-term trends from a signal to isolate short-term features  
* Epoch: The timestamp of the first observed transit centre, as a reference point for phase folding  
* F𝛽: A training performance metric, where model recall is weighed 𝛽 times more than precision. We use an F2 score for our model.  
* FC: Fully Connected  
* FFI: Full-Frame Image  
* FITS: Flexible Image Transport System  
* FLICKER: Fast Light-curve Inference and Classification for Keplerian Exoplanet Recognition  
* Flux: The rate at which light energy is received per unit area  
* Ingress/Egress: The beginning and end of a transit event, respectively, where the planet begins to cross and then clears the stellar disk  
* Logit: The raw, unnormalized output of a network layer before any activation function is applied to convert it to a probability  
* LS: Least Squares, a method to minimize loss compared to a base function  
* MIT: Massachusetts Institute of Technology  
* ML: Machine Learning  
* NASA: National Aeronautics and Space Administration  
* NN: Neural Network  
* Phase Folding: The process of collapsing a time series into a single representative orbit by mapping each timestamp into its position within 1 period  
* Precision: The rate at which a network’s positive classifications are actually correct. Scored from 0 to 1, where 1 is perfect.  
* QLP: Quick-Look Pipeline  
* RA: Right Ascension  
* Recall: The rate at which a network correctly identifies the positive class. Scored from 0 to 1, where 1 is perfect.  
* ReLU: Rectified Linear Unit  
* TCE: Threshold Crossing Event  
* TESS: Transiting Exoplanet Survey Satellite  
* TIC: TESS Input Catalogue  
* TLS: Trapezoidal Least Squares

Scientific notation quick reference:

* $10^{-6}$ \= 0.000001
* $10^{-3}$ \= 0.001
* $10^{-1}$ \= 0.1 \= 1 per ten, s$^{-1}$ \= 1 per second, etc.  
* $10^0$ \= 1  
* $10^1$ \= 10  
* $10^3$ \= 1,000  
* $10^6$ \= 1,000,000
