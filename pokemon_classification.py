import kagglehub
import torch
from torchvision import datasets, transforms, models
from torchvision.models import AlexNet_Weights, ResNet18_Weights, VGG16_Weights, EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class PokemonClassification:

    # 포켓몬 이미지 데이터셋
    dataset = None
    train_dataset, test_dataset = None, None
    num_classes: int

    # 모델
    models_dict: dict = {}

    def __init__(self):
        self.download_and_load_dataset()
        self.build_all_models()

    def run(self):
        st.header("Pokemon Classification")

        # 버튼 눌러 시작
        if not st.button("시작하기"):
            st.info("먼저 '시작하기' 버튼을 눌러주세요.")
            return

        # 모델 준비
        with st.expander("모델 준비", expanded=True):
            for model_name in self.models_dict:
                with st.spinner(f"'{model_name}' 모델 준비 중..."):
                    is_loaded = self.load_or_train_model_and_register(model_name)

                if is_loaded:
                    st.write(f"💾 '{model_name}' 모델 데이터 불러오기 완료")
                else:
                    st.write(f"✅ '{model_name}' 모델 학습 및 평가 완료")

                self.render_learning_curve_plot(model_name)

        # 모델 학습 결과
        with st.expander("모델 평가 결과", expanded=True):
            self.render_model_evaluation_table()

        # 직접 이미지를 추가하여 모델의 결과 확인
        # TODO

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

    # === 모델 빌드 ===

    def build_alexnet(self, fine_tuning: bool = False):
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, self.num_classes)
        if not fine_tuning:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[6].parameters():
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

    def build_resnet18(self, fine_tuning: bool = False):
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
        if not fine_tuning:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
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

    def build_all_models(self):
        self.models_dict["AlexNet"] = self.build_alexnet()
        self.models_dict["AlexNet (fine-tuning)"] = self.build_alexnet(True)
        self.models_dict["VGG16"] = self.build_vgg16()
        self.models_dict["VGG16 (fine-tuning)"] = self.build_vgg16(True)
        self.models_dict["ResNet18"] = self.build_resnet18()
        self.models_dict["ResNet18 (fine-tuning)"] = self.build_resnet18(True)
        self.models_dict["EfficientNet_B0"] = self.build_efficientnet_b0()
        self.models_dict["EfficientNet_B0 (fine-tuning)"] = self.build_efficientnet_b0(True)

    def load_or_train_model_and_register(self, model_name: str) -> bool:
        """
        모델 관련 데이터를 불러온다. 저장된 데이터가 없으면 새로 학습 후 파일로 저장한다. 이후 st.session_state에 저장한다.

        Args:
            model_name (str): 모델 이름
        Returns:
            valid (bool): 저장된 모델 데이터를 성공적으로 불러왔으면 True, 새로 학습했다면 False
        """
        # 세션 상태 초기화
        if model_name not in st.session_state:
            st.session_state[model_name] = {}

        # 저장된 모델 관련 데이터 불러오기
        valid, model, learning_curve, evaluation_results = self.load_saved_model_data(model_name)

        if not valid:  # 모델을 새로 학습 및 관련 데이터 구하기
            model = self.models_dict[model_name]
            learning_curve = {}
            evaluation_results = {}

            # 모델 학습
            progress = st.progress(0, text=f"{model_name} 학습 진행 중...")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.train()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
            criterion = torch.nn.CrossEntropyLoss()
            loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
            total_batches = len(loader)
            num_epochs = 5
            train_losses = []
            train_accuracies = []
            for epoch in range(num_epochs):  # epoch
                epoch_loss = 0
                correct = 0
                total = 0
                for batch_idx, (images, labels) in enumerate(loader, 1):  # batch
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    self.render_train_progress(model_name, epoch, num_epochs, batch_idx, total_batches, progress)
                train_losses.append(epoch_loss / total_batches)
                train_accuracies.append(correct / total)

            progress.empty()

            # 러닝커브
            learning_curve["loss"] = train_losses
            learning_curve["accuracy"] = train_accuracies

            # 모델 평가
            evaluation_results = self.evaluate_model(model)

            # 파일로 저장
            self.save_model(model, model_name)
            self.save_learning_curve(learning_curve, model_name)
            self.save_evaluation_results(evaluation_results, model_name)

        # 세션에 저장
        st.session_state[model_name]["model"] = model
        st.session_state[model_name]["learning_curve"] = learning_curve
        st.session_state[model_name]["evaluation_results"] = evaluation_results

        return valid

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

    # === 데이터 저장/로드 ===

    # 모델을 파일로 저장
    def save_model(self, model, model_name: str):
        model_path = f"./model/{model_name}.pth"
        torch.save(model, model_path)

    # 저장된 모델 로드
    def load_model(self, model_name: str):
        model_path = f"./model/{model_name}.pth"
        return torch.load(model_path, weights_only=False)

    # Learning Curve 데이터 저장
    def save_learning_curve(self, learning_curve: dict, model_name: str):
        curve_path = f"./model/{model_name}_curve.npy"
        np.save(curve_path, learning_curve)

    # Learning Curve 데이터 로드
    def load_learning_curve(self, model_name: str) -> dict:
        curve_path = f"./model/{model_name}_curve.npy"
        return np.load(curve_path, allow_pickle=True).item()

    # 모델 평가 결과 저장
    def save_evaluation_results(self, eval_results: dict, model_name: str):
        eval_path = f"./model/{model_name}_evaluation.npy"
        np.save(eval_path, eval_results)

    # 모델 평가 결과 로드
    def load_evaluation_results(self, model_name: str) -> dict:
        eval_path = f"./model/{model_name}_evaluation.npy"
        return np.load(eval_path, allow_pickle=True).item()

    def load_saved_model_data(self, model_name: str) -> tuple[bool, torch.nn.Module | None, dict | None, dict | None]:
        """
        저장된 학습 관련 데이터(모델, 러닝커브, 평가결과)를 불러온다.

        Args:
            model_name (str): 모델 이름

        Returns:
            tuple:
                - valid (bool): 모든 데이터가 정상적으로 로드되었는지 여부
                - model (torch.nn.Module | None): 저장된 모델 객체 (없으면 None)
                - learning_curve (dict | None): 러닝커브 데이터 (없으면 None)
                - evaluation_results (dict | None): 평가 결과 데이터 (없으면 None)
        """
        valid = True
        model, learning_curve, evaluation_results = None, None, None

        try:
            model = self.load_model(model_name)
            learning_curve = self.load_learning_curve(model_name)
            evaluation_results = self.load_evaluation_results(model_name)
        except Exception as e:
            valid = False

        return valid, model, learning_curve, evaluation_results

    # === UI ===

    def render_train_progress(self, model_name: str, epoch: int, num_epochs: int, batch_idx: int, total_batches: int, container: DeltaGenerator = None):
        container = container if container is not None else st

        container.progress(
            (epoch * total_batches + batch_idx) / (num_epochs * total_batches),
            text=f"{model_name} {epoch+1}/{num_epochs} epoch, {batch_idx}/{total_batches} 배치 학습 완료",
        )

    def render_learning_curve_plot(self, model_name):
        learning_curve = st.session_state[model_name]["learning_curve"]

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")
        ax1.set_xticks(range(len(learning_curve["loss"])))

        ax1.plot(learning_curve["loss"], label="Loss", color="tab:orange")
        ax1.set_ylabel("Loss", color="tab:orange")

        ax2 = ax1.twinx()

        ax2.plot(learning_curve["accuracy"], label="Accuracy", color="tab:blue")
        ax2.set_ylabel("Accuracy", color="tab:blue")

        fig.suptitle(f"{model_name} Learning Curve")
        with st.popover(f"Learning Curve Graph - {model_name}"):
            st.pyplot(fig)

    def render_model_evaluation_table(self):
        with st.spinner("모델 평가 중..."):
            columns = ["model", "accuracy", "precision", "recall", "f1"]
            rows = []
            for model_name in self.models_dict:
                result = st.session_state[model_name]["evaluation_results"]
                rows.append(
                    [
                        model_name,
                        f"{result['accuracy']:.4f}",
                        f"{result['precision']:.4f}",
                        f"{result['recall']:.4f}",
                        f"{result['f1']:.4f}",
                    ]
                )

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
