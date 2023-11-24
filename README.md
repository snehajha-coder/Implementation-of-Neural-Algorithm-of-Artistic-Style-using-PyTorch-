# Implementation-of-Neural-Algorithm-of-Artistic-Style-using-PyTorch
The code is the implementation of Neural-Style algorithm developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge. I have just used tutorial from Py Touch to implement the given task

## What does this model do?
![image](https://github.com/snehajha-coder/Implementation-of-Neural-Algorithm-of-Artistic-Style-using-PyTorch/assets/84180023/120263f8-9c56-45ad-a060-c43ebe0a6d92)
Style of one image is imposed on the other image


## How?

### Data Preprocessing
The input images, both content, and style images are loaded and transformed into PyTorch tensors. The images are resized to a fixed size of 512x512 pixels. The loader function and transformations ensure compatibility with the neural network model.

```python
# Loader function for images
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Load style and content images
style_img = image_loader("style.jpeg")
content_img = image_loader("content.jpeg")

# Check the size of the images
print("Style Image Size:", style_img.size())
print("Content Image Size:", content_img.size())
```

### Model Architecture
The model architecture is based on the VGG19 Convolutional Neural Network, a deep neural network with 19 layers. The model is used to extract content and style representations from the input images. Content and style losses are computed using these representations.
![image](https://github.com/snehajha-coder/Implementation-of-Neural-Algorithm-of-Artistic-Style-using-PyTorch/assets/84180023/324e435e-db99-48a1-b12f-260d3a5a1ace)
![image](https://github.com/snehajha-coder/Implementation-of-Neural-Algorithm-of-Artistic-Style-using-PyTorch/assets/84180023/4f4755a5-9a9a-4a1b-bd0d-7db1e5543398)

```python
# Download pre-trained VGG19 model
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
```

### Training
#### Model Overview:

The Neural Algorithm of Artistic Style, based on the paper by Leon Gatys et al., utilizes a pre-trained Convolutional Neural Network (CNN) like VGG for image classification.

1. **Content and Style Representation:**
   - Content is captured by high-layer activations, representing complex structures.
   - Style is encoded by correlations between activations, capturing textures and patterns.

2. **Loss Function:**
   - Content loss measures differences in content between generated and target images.
   - Style loss measures differences in style between generated and reference images.
   - Total loss is a weighted sum of content and style losses.

#### Optimization Process:

1. **Initialization:**
   - Start with an initialized image, often random noise or a copy of the content image.

2. **Forward Pass:**
   - Compute content and style representations by passing the generated image through the pre-trained network.

3. **Loss Computation:**
   - Calculate content loss by comparing higher layer activations.
   - Compute style loss by comparing correlations between activations.
   - Total loss is a weighted sum of content and style losses.

4. **Backward Pass:**
   - Compute gradients of total loss with respect to pixel values.

5. **Update Image:**
   - Update pixel values in the direction that minimizes the total loss.

6. **Iteration:**
   - Repeat steps 2-5 iteratively until the image converges, blending content and style.


### Evaluation
![image](https://github.com/snehajha-coder/Implementation-of-Neural-Algorithm-of-Artistic-Style-using-PyTorch/assets/84180023/4d790625-24ca-4a43-bb91-a03021f30257)
![image](https://github.com/snehajha-coder/Implementation-of-Neural-Algorithm-of-Artistic-Style-using-PyTorch/assets/84180023/2d1db5d0-b139-4b0d-a644-fed9e33fc9d8)



## Limitations and Potential Improvements

### Limitations
1. **Computational Intensity**: The optimization process is computationally intensive and time-consuming.
2. **Single Style Transfer**: The model is designed for one-to-one style transfer.
3. **Dependence on Pre-trained Models**: Effectiveness relies on the quality of the pre-trained CNN.
4. **Lack of Semantic Control**: The model does not explicitly consider semantic content.
5. **Parameter Sensitivity**: Performance is sensitive to hyperparameters.
6. **Limited Style Understanding**: The model may struggle with intricate local styles.
7. **Style and Content Mismatch**: Balancing content and style can be challenging.
8. **Artifacts and Over-smoothing**: Optimization process might introduce artifacts.
9. **Need for Reference Image**: Style transfer requires a reference image.
10. **Limited Transferability Across Domains**: The model may not perform well with drastically different content.

### Potential Improvements
1. **Optimization Techniques**: Explore alternative optimization algorithms for faster convergence.
2. **Multi-Style Transfer**: Enhance the model to handle multiple style transfers in a single image.
3. **Fine-Tuning**: Fine-tune the model on specific datasets for improved performance.
4. **Semantic Integration**: Integrate semantic information for better content understanding.
5. **Hyperparameter Tuning**: Conduct thorough hyperparameter tuning for optimal results.
6. **Local Style Recognition**: Improve the model's ability to capture intricate local styles.
7. **Dynamic Weighting**: Implement dynamic weighting of style and content losses based on image characteristics.
8. **Artifact Reduction**: Apply techniques to reduce artifacts introduced during the optimization process.
9. **Reference-Free Style Transfer**: Investigate methods for style transfer without the need for a separate reference image.
10. **Domain Adaptation**: Explore techniques for improving transferability across diverse content domains.

**Note:** The model's limitations and potential improvements are crucial for understanding its current constraints and areas where enhancements can be made. Continuous experimentation and research can contribute to addressing these limitations and advancing the capabilities of the Style Capturer.

