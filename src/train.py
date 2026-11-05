import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# chequear dispositivo
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("PyTorch:", torch.__version__, "- Dispositivo:", device)
torch.manual_seed(42)

# cargar datos
df = pd.read_csv('data/iris.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# pasar a tensores
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

# modelo
modelo = nn.Sequential(
    nn.Linear(4, 32),  # 32 neuronas en la capa oculta
    nn.ReLU(),
    nn.Linear(32, 3)
).to(device)

criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.01)

epocas = 100
perdidas_train = []
perdidas_val = []
accuracys_train = []
accuracys_val = []

# entrenamiento
for epoca in range(epocas):
    modelo.train()
    
    # reinicio los gradientes, si no se acumulan
    optimizador.zero_grad()
    
    salida = modelo(X_train)
    perdida = criterio(salida, y_train)
    perdida.backward()
    optimizador.step()
    
    acc_train = (salida.argmax(dim=1) == y_train).float().mean().item()
    
    # en validación no hace falta calcular gradientes
    modelo.eval()
    with torch.no_grad():
        salida_val = modelo(X_val)
        perdida_val = criterio(salida_val, y_val)
        acc_val = (salida_val.argmax(dim=1) == y_val).float().mean().item()
        
    perdidas_train.append(perdida.item())
    perdidas_val.append(perdida_val.item())
    accuracys_train.append(acc_train)
    accuracys_val.append(acc_val)
    
    if (epoca + 1) % 10 == 0:
        print("epoca", epoca + 1, "- loss:", round(perdida.item(), 4), "- val:", round(perdida_val.item(), 4))
        print("acc train:", round(acc_train, 3), "- acc val:", round(acc_val, 3))

# grafico de la curva de pérdida
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(perdidas_train, label='entrenamiento', color='#ed2603', linewidth=2.5)
ax.plot(perdidas_val, label='validación', color='#2f80ff', linewidth=2.5, linestyle='--')

# sombreado entre las dos curvas
ax.fill_between(range(len(perdidas_train)), perdidas_train, perdidas_val, color='white', alpha=0.08)

# marco la loss final de cada una
ax.scatter(epocas - 1, perdidas_train[-1], color='#ed2603', zorder=5)
ax.scatter(epocas - 1, perdidas_val[-1], color='#2f80ff', zorder=5)

# saco los bordes de arriba y de la derecha
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('época')
ax.set_ylabel('loss')
ax.set_title('Cómo bajó la pérdida durante el entrenamiento')
ax.grid(alpha=0.35)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig('results/loss_curve.png', dpi=150)
print("Entrenamiento listo y gráfico guardado.")