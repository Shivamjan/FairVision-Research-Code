import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, Any, Tuple
import cv2

#configuration
st.set_page_config(
    page_title="CARE-Net | AC-CQC-DD with LeGrad",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    /* Main Background & Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: transparent;
    }
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    /* Header & Subtitle */
    h1 {
        color: #667eea;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2rem;
    }

    /* Custom Classes */
    .metric-card {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 20px;
        color: black !important;
    }

    /* Text Color Fixes */
    h3 {
        color: black !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: black !important;
    }
    
    [data-testid="stMetricValue"] {
        color: black !important;
    }
    
    .stProgress > div > div > div > div {
        color: black !important;
        font-weight: 600;
    }
    
    p {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

class LeGradExplainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def generate_saliency(self, 
                         input_tensor: torch.Tensor, 
                         target_class: int = None,
                         method: str = 'vanilla') -> np.ndarray:
        


        if method == 'vanilla':
            return self._vanilla_gradient(input_tensor, target_class)
        elif method == 'smooth':
            return self._smooth_gradient(input_tensor, target_class)
        elif method == 'integrated':
            return self._integrated_gradients(input_tensor, target_class)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _vanilla_gradient(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        logits, _ = self.model(input_tensor)
        
        if target_class is None:
            target_class = logits.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        

        gradients = input_tensor.grad.data.abs()
        
        saliency = gradients.max(dim=1)[0]  # [1, H, W]
        saliency = saliency.squeeze().cpu().numpy()
        
       
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)  # Normalize
        
        return saliency
    
    def _smooth_gradient(self, input_tensor: torch.Tensor, target_class: int = None, 
                        n_samples: int = 50, noise_level: float = 0.15) -> np.ndarray:



        saliency_maps = []
        
        for _ in range(n_samples):

            noise = torch.randn_like(input_tensor) * noise_level
            noisy_input = input_tensor + noise
            noisy_input = noisy_input.requires_grad_(True)
            

            logits, _ = self.model(noisy_input)
            
            if target_class is None:
                target_class = logits.argmax(dim=1)

            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)

            gradients = noisy_input.grad.data.abs()
            saliency = gradients.max(dim=1)[0].squeeze().cpu().numpy()
            saliency_maps.append(saliency)

        saliency = np.mean(saliency_maps, axis=0)
        
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        
        return saliency
    
    def _integrated_gradients(self, input_tensor: torch.Tensor, target_class: int = None,
                             n_steps: int = 50, baseline: torch.Tensor = None) -> np.ndarray:
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        

        alphas = torch.linspace(0, 1, n_steps).to(self.device)
        integrated_grads = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated = interpolated.requires_grad_(True)
            
            logits, _ = self.model(interpolated)
            
            if target_class is None:
                target_class = logits.argmax(dim=1)
            
            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)
            
            gradients = interpolated.grad.data
            integrated_grads.append(gradients.cpu().numpy())
        
        avg_grads = np.mean(integrated_grads, axis=0)
        integrated_gradients = (input_tensor - baseline).cpu().numpy() * avg_grads
        
        saliency = np.abs(integrated_gradients).max(axis=1)[0]
        
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        
        return saliency
    



def create_attention_overlay(image: Image.Image, attention_map: np.ndarray, alpha: float = 0.5) -> Image.Image:
    img_array = np.array(image.resize((224, 224)))
    
    if len(attention_map.shape) == 4:
        attention_map = attention_map[0, 0]
    elif len(attention_map.shape) == 3:
        attention_map = attention_map[0]
    
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-10)
    attention_resized = cv2.resize(attention_map, (224, 224))
    
    heatmap = cm.jet(attention_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    
    return Image.fromarray(overlay)


def create_legrad_overlay(image: Image.Image, saliency_map: np.ndarray, alpha: float = 0.5) -> Image.Image:
    img_array = np.array(image.resize((224, 224)))
    
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-10)
    
    heatmap = cm.hot(saliency_map)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    
    return Image.fromarray(overlay)


def create_comprehensive_visualization(image: Image.Image, 
                                      attention_map: np.ndarray,
                                      legrad_map: np.ndarray,
                                      prediction: str, 
                                      confidence: float):

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    img_resized = image.resize((224, 224))
    
    if len(attention_map.shape) == 4:
        attention_map = attention_map[0, 0]
    elif len(attention_map.shape) == 3:
        attention_map = attention_map[0]
    attention_resized = cv2.resize(attention_map, (224, 224))
    
    axes[0, 0].imshow(img_resized)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(attention_resized, cmap='jet', interpolation='bilinear')
    axes[0, 1].set_title("Model Attention Map", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    overlay_attention = create_attention_overlay(image, attention_map, alpha=0.4)
    axes[0, 2].imshow(overlay_attention)
    axes[0, 2].set_title("Attention Overlay", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_resized)
    axes[1, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(legrad_map, cmap='hot', interpolation='bilinear')
    axes[1, 1].set_title("LeGrad Saliency Map", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    overlay_legrad = create_legrad_overlay(image, legrad_map, alpha=0.4)
    axes[1, 2].imshow(overlay_legrad)
    axes[1, 2].set_title(f"LeGrad Overlay\n{prediction} ({confidence:.1%})", 
                        fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def create_comparison_visualization(image: Image.Image,
                                   attention_map: np.ndarray,
                                   legrad_map: np.ndarray,
                                   prediction: str,
                                   confidence: float):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img_resized = image.resize((224, 224))
    
    if len(attention_map.shape) == 4:
        attention_map = attention_map[0, 0]
    elif len(attention_map.shape) == 3:
        attention_map = attention_map[0]
    attention_resized = cv2.resize(attention_map, (224, 224))
    
    axes[0].imshow(img_resized)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    overlay_attention = create_attention_overlay(image, attention_map, alpha=0.5)
    axes[1].imshow(overlay_attention)
    axes[1].set_title("Model Attention (Internal)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    overlay_legrad = create_legrad_overlay(image, legrad_map, alpha=0.5)
    axes[2].imshow(overlay_legrad)
    axes[2].set_title("LeGrad (Gradient-based)", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f"Prediction: {prediction} ({confidence:.1%})", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


class ACCQCDDInferenceWithLeGrad:
    
    def __init__(self, checkpoint_path: str, num_classes: int = 3, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        from inference import load_model

        self.model = load_model(checkpoint_path, num_classes, str(self.device))
        self.model.eval()
        
        self.legrad_explainer = LeGradExplainer(self.model, self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.num_classes = num_classes
        self.class_names = self._get_class_names(num_classes)
    
    def _get_class_names(self, num_classes: int) -> list:
        if num_classes == 3:
            return ['Benign', 'Malignant', 'Non-Neoplastic']
        else:
            return ['Benign', 'Malignant']
    
    def predict(self, image: Image.Image, legrad_method: str = 'smooth') -> Dict[str, Any]:
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Model prediction
            logits, diagnostics = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
            
            attention_map = diagnostics['attention_weights'].cpu().numpy()
        
        legrad_map = self.legrad_explainer.generate_saliency(
            input_tensor, 
            target_class=pred_class,
            method=legrad_method
        )

        from inference import compute_fairness_metrics
        fairness_metrics = compute_fairness_metrics(diagnostics, probs)
        
        prob_dict = {self.class_names[i]: probs[0, i].item() 
                    for i in range(self.num_classes)}
        
        return {
            'prediction': self.class_names[pred_class],
            'confidence': confidence,
            'probabilities': prob_dict,
            'attention_map': attention_map,
            'legrad_map': legrad_map,
            'fairness_metrics': fairness_metrics,
            'diagnostics': diagnostics
        }

@st.cache_resource
def load_model_cached(checkpoint_path: str, num_classes: int):
    """Cache model loading to avoid reloading on every interaction"""
    return ACCQCDDInferenceWithLeGrad(checkpoint_path, num_classes)


def main():
    st.title("🏥 CARE-Net: Medical Image Analysis")
    st.markdown('<p class="subtitle">AC-CQC-DD Model with LeGrad Explainability</p>', 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.selectbox(
            "Classification Mode",
            options=["3-Class", "2-Class"],
            help="3-Class: Benign/Malignant/Non-Neoplastic\n2-Class: Benign/Malignant"
        )
        num_classes = 3 if mode == "3-Class" else 2
        
        st.divider()
        
        st.subheader("Model Configuration")
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="model/weights/ac_cqc_dd_final.pt",
            help="Path to your trained AC-CQC-DD model checkpoint"
        )
        
        st.divider()
        st.subheader("LeGrad Options")
        legrad_method = st.selectbox(
            "Saliency Method",
            options=["smooth", "vanilla", "integrated"],
            help="""
            - Vanilla: Standard gradients (fast, noisy)
            - Smooth: Averaged over noisy samples (balanced)
            - Integrated: Path integration (slow, accurate)
            """
        )
        
        st.divider()
        st.subheader("Visualization Options")
        
        viz_mode = st.radio(
            "Display Mode",
            options=["Comparison", "Comprehensive", "Separate"],
            help="""
            - Comparison: Side-by-side Attention vs LeGrad
            - Comprehensive: 2x3 grid with all visualizations
            - Separate: Individual components in columns
            """
        )
        
        overlay_alpha = st.slider(
            "Heatmap Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        st.divider()
        
        st.info(
            "**About AC-CQC-DD + LeGrad**\n\n"
            "Combines model-internal attention with gradient-based explanations "
            "for comprehensive interpretability."
        )
        
        st.warning(
            "**Research Demo Only**\n\n"
            "Not for clinical use without validation."
        )
    

    st.subheader("Upload Medical Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"],
        help="Upload a histopathology or medical image"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col_upload, col_button = st.columns([3, 1])
        with col_upload:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col_button:
            if st.button("Run Inference", type="primary", use_container_width=True):
                with st.spinner(f"Running AC-CQC-DD with LeGrad ({legrad_method})..."):
                    try:
                        model = load_model_cached(checkpoint_path, num_classes)
                        result = model.predict(image, legrad_method=legrad_method)
                        
                        st.session_state['result'] = result
                        st.session_state['image'] = image
                        
                    except Exception as e:
                        st.error(f"Error during inference: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        st.divider()
        

        if 'result' in st.session_state:
            result = st.session_state['result']
            original_image = st.session_state['image']
            
            # Prediction header
            st.subheader(" Prediction Results")
            
            pred_color = {
                'Benign': '#4caf50',
                'Malignant': '#f44336',
                'Non-Neoplastic': '#2196f3'
            }
            color = pred_color.get(result['prediction'], '#666')
            
            st.markdown(f"""
                <div class='prediction-box' style='background: {color};'>
                    Predicted: {result['prediction']} (Confidence: {result['confidence']:.1%})
                </div>
            """, unsafe_allow_html=True)
            

            st.subheader(" Explainability Visualization")
            
            if viz_mode == "Comparison":
                fig = create_comparison_visualization(
                    original_image,
                    result['attention_map'],
                    result['legrad_map'],
                    result['prediction'],
                    result['confidence']
                )
                st.pyplot(fig)
                
            elif viz_mode == "Comprehensive":
                fig = create_comprehensive_visualization(
                    original_image,
                    result['attention_map'],
                    result['legrad_map'],
                    result['prediction'],
                    result['confidence']
                )
                st.pyplot(fig)
                
            else:  # Separate
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(original_image.resize((224, 224)), caption="Original")
                
                with col2:
                    st.markdown("**Model Attention**")
                    overlay_att = create_attention_overlay(
                        original_image, 
                        result['attention_map'], 
                        alpha=overlay_alpha
                    )
                    st.image(overlay_att, caption="Attention Overlay")
                
                with col3:
                    st.markdown("**LeGrad Saliency**")
                    overlay_leg = create_legrad_overlay(
                        original_image,
                        result['legrad_map'],
                        alpha=overlay_alpha
                    )
                    st.image(overlay_leg, caption="LeGrad Overlay")
            
            st.divider()
            
            # Metrics Section
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader(" Class Probabilities")
                for class_name, prob in result['probabilities'].items():
                    st.progress(prob, text=f"{class_name}: {prob:.1%}")
            
            with col_right:
                st.subheader("⚖️ Fairness Diagnostics")
                metrics = result['fairness_metrics']
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("CF Consistency", f"{metrics['counterfactual_consistency']:.1%}")
                    st.metric("Quality Score", f"{metrics['quality_score']:.1%}")
                
                with col_b:
                    st.metric("Attention Entropy", f"{metrics['attention_entropy']:.2f}")
                    st.metric("Decision Confidence", f"{metrics['decision_confidence']:.1%}")
            
            # Interpretation
            st.divider()
            st.subheader("Interpretation")
            
            col_int1, col_int2 = st.columns(2)
            
            with col_int1:
                st.markdown("**Model Attention vs LeGrad:**")
                st.markdown("""
                - **Attention (Jet colormap)**: Shows where the model *internally focuses* during forward pass
                - **LeGrad (Hot colormap)**: Shows which pixels *contribute most* to the prediction via gradients
                - **Agreement**: High overlap = robust, interpretable decision
                - **Disagreement**: May indicate reliance on spurious features or texture bias
                """)
            
            with col_int2:
                if metrics['counterfactual_consistency'] > 0.9:
                    st.markdown(":green[**High CF Consistency** - Robust prediction]")
                elif metrics['counterfactual_consistency'] > 0.7:
                    st.markdown(":orange[**Moderate Consistency** - Some uncertainty]")
                else:
                    st.markdown(":red[**Low Consistency** - Review carefully]")
                
                if metrics['attention_entropy'] < 0.5:
                    st.markdown(":green[**Focused Attention** - Specific regions]")
                else:
                    st.markdown(":orange[**Diffuse Attention** - Spread across image]")
    
    else:
        st.info("Upload an image to begin analysis")
    
    # Additional information
    st.divider()
    
    with st.expander("About This System"):
        st.markdown("""
        ### AC-CQC-DD + LeGrad Integration
        
        **Model Components:**
        - **AC-CQC-DD**: Attention-based Counterfactual Consistency with Quality-aware Decision making
        - **LeGrad**: Learning to Generate Gradient-based explanations
        
        **Explainability Methods:**
        
        1. **Model Attention (Built-in)**
           - Extracted from internal attention mechanisms
           - Shows which image patches the model attends to
           - Computed during forward pass (no extra computation)
        
        2. **LeGrad Saliency (Gradient-based)**
           - **Vanilla**: Direct gradients w.r.t. input pixels
           - **SmoothGrad**: Averaged over noisy samples (reduces noise)
           - **Integrated Gradients**: Path integration from baseline (most principled)
        
        **Why Both?**
        - Attention shows *internal focus* (where model looks)
        - LeGrad shows *pixel importance* (what drives the decision)
        - Complementary insights improve trust and debugging
        
        **Color Schemes:**
        - 🔴 Red/Yellow (Jet): High attention/importance
        - 🔵 Blue/Purple: Low attention/importance
        - 🟠 Hot colormap: LeGrad intensity
        """)
    
    with st.expander("How to Use"):
        st.markdown("""
        1. **Select classification mode** (2-class or 3-class)
        2. **Choose LeGrad method**:
           - Vanilla: Fastest, may be noisy
           - Smooth: Recommended for most cases
           - Integrated: Most accurate, slowest
        3. **Upload medical image**
        4. **Click 'Run Inference'**
        5. **Compare explanations**:
           - Look for agreement between attention and LeGrad
           - Disagreement may indicate spurious correlations
        6. **Check fairness metrics** for reliability
        
        **Tips:**
        - Use 'Comparison' mode to quickly compare methods
        - Adjust opacity to see overlay details
        - High CF consistency + focused attention = reliable prediction
        """)
    
    with st.expander("References"):
        st.markdown("""
        **LeGrad Paper:**
        - Title: *Learning to Generate Gradient for Medical Image Diagnosis Interpretation*
        - Key idea: Learn to weight gradients from different layers for better explanations
        
        **Related Methods:**
        - Grad-CAM: Class Activation Mapping
        - SmoothGrad: Reduces gradient noise via averaging
        - Integrated Gradients: Path-based attribution
        
        **Your Model:**
        - AC-CQC-DD with GQVK architecture
        - Vision Transformer backbone with learned queries
        - Counterfactual consistency training
        """)


if __name__ == "__main__":
    main()
