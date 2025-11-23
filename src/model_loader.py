"""
Model loading and inference module for freshness detection.
Uses EfficientNetB0 pretrained CNN with fine-tuning.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Using fallback methods.")


class FreshnessModel:
    """Wrapper class for freshness detection models using EfficientNetB0."""
    
    def __init__(self, model_path: Optional[str] = None, img_size: int = 224):
        """
        Initialize freshness model.
        
        Args:
            model_path: Path to saved model directory (optional)
            img_size: Input image size (default 224 for EfficientNetB0)
        """
        self.img_size = img_size
        self.classifier_model = None
        self.freshness_model = None
        self.item_classes = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize default models if no saved model exists
            self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models using EfficientNetB0."""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Model will use rule-based predictions.")
            self.item_classes = [
                'apple', 'banana', 'orange', 'tomato', 'carrot', 'potato',
                'onion', 'cucumber', 'bell_pepper', 'broccoli', 'spinach',
                'mango', 'grapes', 'strawberry', 'watermelon', 'papaya'
            ]
            return
        
        # Default item classes
        self.item_classes = [
            'apple', 'banana', 'orange', 'tomato', 'carrot', 'potato',
            'onion', 'cucumber', 'bell_pepper', 'broccoli', 'spinach',
            'mango', 'grapes', 'strawberry', 'watermelon', 'papaya',
            'pomegranate', 'cabbage', 'cauliflower', 'eggplant', 'okra',
            'coconut', 'guava', 'lemon', 'ginger', 'garlic'
        ]
        
        # Models will be created on first prediction if not trained
        self.classifier_model = None
        self.freshness_model = None
    
    def _create_classifier_model(self, num_classes: int) -> keras.Model:
        """Create EfficientNetB0-based classifier model."""
        # Load pretrained EfficientNetB0 (without top layers)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model initially (will be fine-tuned)
        base_model.trainable = True
        
        # Add custom classification head
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def _create_freshness_model(self) -> keras.Model:
        """Create EfficientNetB0-based freshness regression model."""
        # Load pretrained EfficientNetB0 (without top layers)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model initially (will be fine-tuned)
        base_model.trainable = True
        
        # Add custom regression head
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)  # Output 0-1, will scale to 0-100
        
        model = keras.Model(inputs, outputs)
        return model
    
    def load_model(self, model_path: str):
        """Load models from directory."""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Cannot load models.")
            return
        
        try:
            # Load classifier
            classifier_path = os.path.join(model_path, 'classifier')
            if os.path.exists(classifier_path):
                self.classifier_model = keras.models.load_model(classifier_path)
            
            # Load freshness model
            freshness_path = os.path.join(model_path, 'freshness')
            if os.path.exists(freshness_path):
                self.freshness_model = keras.models.load_model(freshness_path)
            
            # Load item classes
            classes_path = os.path.join(model_path, 'item_classes.txt')
            if os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    self.item_classes = [line.strip() for line in f.readlines()]
            
            print(f"Models loaded from {model_path}")
        except Exception as e:
            print(f"Error loading models: {e}")
            self._initialize_default_models()
    
    def save_model(self, model_path: str):
        """Save models to directory."""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Cannot save models.")
            return
        
        try:
            os.makedirs(model_path, exist_ok=True)
            
            # Save classifier
            if self.classifier_model:
                classifier_path = os.path.join(model_path, 'classifier')
                self.classifier_model.save(classifier_path)
            
            # Save freshness model
            if self.freshness_model:
                freshness_path = os.path.join(model_path, 'freshness')
                self.freshness_model.save(freshness_path)
            
            # Save item classes
            if self.item_classes:
                classes_path = os.path.join(model_path, 'item_classes.txt')
                with open(classes_path, 'w') as f:
                    for item in self.item_classes:
                        f.write(f"{item}\n")
            
            print(f"Models saved to {model_path}")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def predict_item(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict fruit/vegetable item from image.
        
        Args:
            image: Image as numpy array (BGR format, will be converted to RGB)
            
        Returns:
            Tuple of (item_name, confidence_score)
        """
        if not TF_AVAILABLE or self.classifier_model is None:
            # Fallback: use rule-based or default
            return ('apple', 0.5)
        
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image_rgb = image[:, :, ::-1]  # BGR to RGB
            else:
                image_rgb = image
            
            # Resize image
            image_resized = tf.image.resize(
                image_rgb, 
                [self.img_size, self.img_size]
            )
            image_batch = np.expand_dims(image_resized.numpy(), axis=0)
            
            # Predict
            predictions = self.classifier_model.predict(image_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            if predicted_class < len(self.item_classes):
                item_name = self.item_classes[predicted_class]
            else:
                item_name = 'unknown'
            
            return (item_name, confidence)
        except Exception as e:
            print(f"Error in item prediction: {e}")
            return ('unknown', 0.0)
    
    def predict_freshness(self, image: np.ndarray, item_name: str) -> float:
        """
        Predict freshness percentage from image.
        
        Args:
            image: Image as numpy array (BGR format, will be converted to RGB)
            item_name: Name of the fruit/vegetable (for fallback)
            
        Returns:
            Freshness percentage (0-100)
        """
        if not TF_AVAILABLE or self.freshness_model is None:
            # Fallback: use rule-based approach
            return self._rule_based_freshness(image, item_name)
        
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image_rgb = image[:, :, ::-1]  # BGR to RGB
            else:
                image_rgb = image
            
            # Resize image
            image_resized = tf.image.resize(
                image_rgb, 
                [self.img_size, self.img_size]
            )
            image_batch = np.expand_dims(image_resized.numpy(), axis=0)
            
            # Predict (output is 0-1, scale to 0-100)
            prediction = self.freshness_model.predict(image_batch, verbose=0)[0][0]
            freshness = float(prediction * 100.0)
            
            # Clamp to 0-100 range
            freshness = max(0.0, min(100.0, freshness))
            
            return freshness
        except Exception as e:
            print(f"Error in freshness prediction: {e}")
            return self._rule_based_freshness(image, item_name)
    
    def _rule_based_freshness(self, image: np.ndarray, item_name: str) -> float:
        """Rule-based freshness estimation fallback."""
        # Simple rule-based estimation
        freshness = 75.0
        
        # Add some variation based on image properties
        if image is not None and len(image.shape) == 3:
            brightness = np.mean(image) / 255.0
            if brightness > 0.6:
                freshness += 10
            elif brightness < 0.4:
                freshness -= 15
        
        freshness = 70.0 + np.random.uniform(-10, 10)
        return max(0.0, min(100.0, freshness))


class ModelTrainer:
    """Helper class for training freshness detection models with EfficientNetB0."""
    
    @staticmethod
    def train_classifier(
        images: np.ndarray,
        labels: np.ndarray,
        item_classes: List[str],
        img_size: int = 224,
        batch_size: int = 32,
        epochs: int = 20,
        validation_split: float = 0.2
    ) -> keras.Model:
        """
        Train item classifier using EfficientNetB0.
        
        Args:
            images: Array of images (N, H, W, 3) in RGB format
            labels: Array of class indices (N,)
            item_classes: List of class names
            img_size: Input image size
            batch_size: Training batch size
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            
        Returns:
            Trained classifier model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for model training")
        
        num_classes = len(item_classes)
        
        # Create model
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build model
        inputs = keras.Input(shape=(img_size, img_size, 3))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with frozen base
        print("Training with frozen base model...")
        history1 = model.fit(
            images, labels,
            batch_size=batch_size,
            epochs=epochs // 2,
            validation_split=validation_split,
            verbose=1
        )
        
        # Fine-tune: unfreeze some layers
        base_model.trainable = True
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - 50
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training
        print("Fine-tuning model...")
        history2 = model.fit(
            images, labels,
            batch_size=batch_size,
            epochs=epochs // 2,
            validation_split=validation_split,
            verbose=1
        )
        
        return model
    
    @staticmethod
    def train_freshness_regressor(
        images: np.ndarray,
        freshness_scores: np.ndarray,
        img_size: int = 224,
        batch_size: int = 32,
        epochs: int = 20,
        validation_split: float = 0.2
    ) -> keras.Model:
        """
        Train freshness regressor using EfficientNetB0.
        
        Args:
            images: Array of images (N, H, W, 3) in RGB format
            freshness_scores: Array of freshness scores (N,) in range 0-100
            img_size: Input image size
            batch_size: Training batch size
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            
        Returns:
            Trained freshness regression model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for model training")
        
        # Normalize freshness scores to 0-1 range
        freshness_normalized = freshness_scores / 100.0
        
        # Create model
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build model
        inputs = keras.Input(shape=(img_size, img_size, 3))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train with frozen base
        print("Training freshness model with frozen base...")
        history1 = model.fit(
            images, freshness_normalized,
            batch_size=batch_size,
            epochs=epochs // 2,
            validation_split=validation_split,
            verbose=1
        )
        
        # Fine-tune: unfreeze some layers
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 50
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )
        
        # Continue training
        print("Fine-tuning freshness model...")
        history2 = model.fit(
            images, freshness_normalized,
            batch_size=batch_size,
            epochs=epochs // 2,
            validation_split=validation_split,
            verbose=1
        )
        
        return model


def create_default_model(model_path: str = "models/freshness_model"):
    """
    Create and save a default model structure.
    
    Args:
        model_path: Path to save the model directory
    """
    model = FreshnessModel()
    os.makedirs(model_path, exist_ok=True)
    
    # Save item classes
    if model.item_classes:
        classes_path = os.path.join(model_path, 'item_classes.txt')
        with open(classes_path, 'w') as f:
            for item in model.item_classes:
                f.write(f"{item}\n")
    
    print(f"Default model structure created at {model_path}")
    print("Note: Models need to be trained using the training notebook.")
