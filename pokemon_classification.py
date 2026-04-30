import kagglehub
import torch
from torchvision import datasets, transforms, models
from torchvision.models import AlexNet_Weights, ResNet18_Weights, VGG16_Weights, EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import pandas as pd
import numpy as np


class PokemonClassification:

    # 포켓몬 이미지 데이터셋
    dataset = None
    train_dataset, test_dataset = None, None
    num_classes: int

    def __init__(self):
        self.download_and_load_dataset()

    def run(self):
        st.header("Pokemon Classification")

        # 모델 얻기 (로드/학습)
        if st.button("모델 불러오기/학습하기"):
            # 각 모델 빌드
            models = {}
            with st.expander("모델 학습", expanded=True):
                models["AlexNet"] = self.get_model_with_ui("AlexNet", self.build_alexnet)
                models["ResNet18"] = self.get_model_with_ui("ResNet18", self.build_resnet18)
                models["VGGNet16"] = self.get_model_with_ui("VGGNet16", self.build_vgg16)
                models["EfficientNet-B0"] = self.get_model_with_ui("EfficientNet-B0", self.build_efficientnet_b0)

                models["AlexNet (fine-tuning)"] = self.get_model_with_ui("AlexNet (fine-tuning)", self.build_alexnet, True)
                models["ResNet18 (fine-tuning)"] = self.get_model_with_ui("ResNet18 (fine-tuning)", self.build_resnet18, True)
                models["VGGNet16 (fine-tuning)"] = self.get_model_with_ui("VGGNet16 (fine-tuning)", self.build_vgg16, True)
                models["EfficientNet-B0 (fine-tuning)"] = self.get_model_with_ui("EfficientNet-B0 (fine-tuning)", self.build_efficientnet_b0, True)

            st.session_state["models"] = models  # 세션에 저장

        # 모델 평가
        if "models" in st.session_state:
            with st.expander("모델 평가", expanded=True):
                with st.spinner("모델 평가 중..."):
                    models = st.session_state["models"]
                    total_models = len(models)
                    progress = st.progress(0, text="모델 평가 진행 중...")

                    rows = []
                    columns = ["model", "accuracy", "precision", "recall", "f1"]
                    for idx, (name, model) in enumerate(models.items(), 1):
                        metrics = self.evaluate_model(model)
                        rows.append(
                            [
                                name,
                                f"{metrics['accuracy']:.4f}",
                                f"{metrics['precision']:.4f}",
                                f"{metrics['recall']:.4f}",
                                f"{metrics['f1']:.4f}",
                            ]
                        )
                        progress.progress(idx / total_models, text=f"{idx}/{total_models} 모델 평가 완료")
                    progress.empty()
                    df = pd.DataFrame(rows, columns=columns)
                    st.dataframe(
                        df,
                        column_config={
                            "model": "모델",
                            "accuracy": "정확도(Accuracy)",
                            "precision": "정밀도(Precision)",
                            "recall": "재현율(Recall)",
                            "f1": "F1-score",
                        },
                        hide_index=True,
                    )
        else:
            st.info("먼저 '모델 불러오기/학습하기' 버튼을 눌러주세요.")

    # 데이터셋 다운로드 및 로드
    def download_and_load_dataset(self):
        # 데이터 다운로드
        path = kagglehub.dataset_download("lantian773030/pokemonclassification", output_dir="./dataset")
        # 데이터셋 로드
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])  # 224x224
        self.dataset = datasets.ImageFolder(f"{path}/PokemonData", transform=transform)
        self.num_classes = len(self.dataset.classes)
        # 학습/테스트 데이터셋 분할 (80:20)
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

    # Pretrained Model 빌드
    def build_alexnet(self, fine_tuning: bool = False):
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, self.num_classes)
        if not fine_tuning:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[6].parameters():
                param.requires_grad = True
        return model

    def build_resnet18(self, fine_tuning: bool = False):
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
        if not fine_tuning:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model

    def build_vgg16(self, fine_tuning: bool = False):
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, self.num_classes)
        if not fine_tuning:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[6].parameters():
                param.requires_grad = True
        return model

    def build_efficientnet_b0(self, fine_tuning: bool = False):
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, self.num_classes)
        if not fine_tuning:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[1].parameters():
                param.requires_grad = True
        return model

    def get_model_with_ui(self, model_name, build_function, fine_tuning: bool = False):
        try:
            # 저장된 모델이 있다면 불러오기
            model = self.load_model(model_name)
            st.write(f"💾 저장된 '{model_name}' 모델을 불러왔습니다.")
            return model
        except Exception as e:
            # 모델을 새로 학습하기
            st.warning(f"저장된 모델을 불러올 수 없어 새로 학습합니다. (사유: {e})")
            with st.spinner(f"'{model_name}' 모델 학습 중..."):
                model = build_function(fine_tuning)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.train()
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
                criterion = torch.nn.CrossEntropyLoss()
                loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
                total_batches = len(loader)
                progress = st.progress(0, text=f"{model_name} 학습 진행 중...")
                for batch_idx, (images, labels) in enumerate(loader, 1):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    progress.progress(batch_idx / total_batches, text=f"{model_name} {batch_idx}/{total_batches} 배치 학습 완료")
                progress.empty()
            st.write(f"✅ '{model_name}' 모델 학습 완료")
            self.save_model(model, model_name)
            return model

    # 모델을 파일로 저장
    def save_model(self, model, model_name):
        torch.save(model, f"./model/{model_name}.pth")

    # 저장된 모델 불러오기
    def load_model(self, model_name: str):
        return torch.load(f"./model/{model_name}.pth", weights_only=False)

    # 모델 성능 측정 (정확도, 정밀도, 재현율, F1)
    def evaluate_model(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        all_preds = []
        all_labels = []
        loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=256)
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
