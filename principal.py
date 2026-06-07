import sys
import os
import shutil
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QLabel, QLineEdit,
                             QPushButton, QTextEdit, QGroupBox, QComboBox,
                             QSpinBox, QGridLayout, QCheckBox, QDialog, QFileDialog)
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt

from unidimensionais.bissecao import bissecao
from unidimensionais.newton import newton
from unidimensionais.wolfe import wolfe
from multidimensionais.fletcher_reeves import fletcher_reeves
from multidimensionais.bfgs import bfgs


class JanelaGif(QDialog):
    """Janela Pop-up para exibir e salvar o GIF gerado"""
    def __init__(self, caminho_gif, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Análise de Convergência (GIF)")
        self.setFixedSize(1250, 600)
        self.caminho_gif = caminho_gif
        
        layout = QVBoxLayout(self)
        
        self.label_animacao = QLabel(self)
        self.label_animacao.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_animacao)
        
        self.filme = QMovie(self.caminho_gif)
        self.label_animacao.setMovie(self.filme)
        self.filme.start()
        
        self.btn_salvar = QPushButton("Salvar GIF no Computador")
        self.btn_salvar.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; font-weight: bold;")
        self.btn_salvar.clicked.connect(self.salvar_gif)
        layout.addWidget(self.btn_salvar)
        
    def salvar_gif(self):
        destino, _ = QFileDialog.getSaveFileName(self, "Salvar Animação", "", "GIF Animado (*.gif)")
        if destino:
            shutil.copy2(self.caminho_gif, destino)
            
    def closeEvent(self, event):
        self.filme.stop()
        try:
            if os.path.exists(self.caminho_gif):
                os.remove(self.caminho_gif)
        except:
            pass
        super().closeEvent(event)


class ModuloOtimizacaoBase(QWidget):
    """Módulo genérico de interface para métodos multidimensionais"""
    def __init__(self, nome_metodo):
        super().__init__()
        self.nome_metodo = nome_metodo
        self.symbols = []
        self.f_sym = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Grupo da função
        func_group = QGroupBox("Configurações da Função")
        func_layout = QGridLayout()
        
        func_layout.addWidget(QLabel("Função f(x):"), 0, 0)
        self.func_input = QLineEdit()
        self.func_input.setPlaceholderText("Ex: x**2 + y**2")
        func_layout.addWidget(self.func_input, 0, 1)
        
        func_layout.addWidget(QLabel("Ponto Inicial (x₀):"), 1, 0)
        self.inicial_input = QLineEdit()
        self.inicial_input.setPlaceholderText("Ex: 1.5 -2.0")
        func_layout.addWidget(self.inicial_input, 1, 1)
        
        func_layout.addWidget(QLabel("Busca do Passo (α):"), 2, 0)
        self.combo_passo = QComboBox()
        self.combo_passo.addItems(["Bisseção", "Newton", "Wolfe"])
        func_layout.addWidget(self.combo_passo, 2, 1)
        
        func_group.setLayout(func_layout)
        layout.addWidget(func_group)
        
        # Grupo de parâmetros
        params_group = QGroupBox("Critérios de Parada e Saída")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Tol. do Gradiente:"), 0, 0)
        params_layout.addWidget(QLabel("Tolerância (10^):"), 0, 0)
        self.tol_input = QSpinBox()
        self.tol_input.setRange(-15, -1)
        self.tol_input.setValue(-4) # Padrão 10^-4
        params_layout.addWidget(self.tol_input, 0, 1)

        # Label para mostrar o valor real
        self.tol_valor_label = QLabel("= 1.00e-04")
        self.tol_valor_label.setStyleSheet("color: gray;")
        params_layout.addWidget(self.tol_valor_label, 0, 2)

        self.tol_input.valueChanged.connect(self.atualizar_label_tolerancia)
        
        params_layout.addWidget(QLabel("Max Iterações:"), 1, 0)
        self.max_iter_input = QSpinBox()
        self.max_iter_input.setRange(1, 2000)
        self.max_iter_input.setValue(100)
        params_layout.addWidget(self.max_iter_input, 1, 1)
        
        self.check_gif = QCheckBox("Gerar GIF Animado (Trajetória + Decaimento)")
        self.check_gif.setChecked(True)
        params_layout.addWidget(self.check_gif, 2, 0, 1, 2)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Botão de Calcular
        self.calc_btn = QPushButton(f"Executar {self.nome_metodo}")
        self.calc_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; font-size: 14px;")
        layout.addWidget(self.calc_btn)
        
        # Log de Resultados
        results_group = QGroupBox("Histórico de Iterações")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.calc_btn.clicked.connect(self.executar_otimizacao)

    def atualizar_label_tolerancia(self):
        """Atualiza a label que mostra o valor real da tolerância"""
        valor = 10 ** self.tol_input.value()
        self.tol_valor_label.setText(f"= {valor:.2e}")

    def executar_otimizacao(self):
        self.results_text.clear()
        try:
            expr_str = self.func_input.text()
            self.f_sym = sp.sympify(expr_str)
            self.symbols = sorted(list(self.f_sym.free_symbols), key=str)
            
            ponto_str = self.inicial_input.text().strip('[]')
            x0 = [float(val) for val in ponto_str.split()]
            
            if len(self.symbols) != len(x0):
                self.results_text.setText("Erro: Número de variáveis na função não bate com o ponto inicial.")
                return

            f_num = sp.lambdify(self.symbols, self.f_sym, 'numpy')
            grad_sym = [sp.diff(self.f_sym, var) for var in self.symbols]
            grad_num = sp.lambdify(self.symbols, grad_sym, 'numpy')
            
            def f_eval(x_arr): return float(f_num(*x_arr))
            def grad_eval(x_arr): return np.array(grad_num(*x_arr), dtype=float)

            metodo_passo = self.combo_passo.currentText()
            
            def calcular_alpha(phi_func, x_k, d_k):
                if metodo_passo == "Bisseção":
                    res = bissecao(phi_func, a=0.0, b=5.0, tol=1e-4)
                    return res['alpha']
                    
                elif metodo_passo == "Wolfe":
                    def phi_prime(alpha):
                        g_val = grad_eval(x_k + alpha * d_k)
                        return float(np.dot(g_val, d_k))
                    res = wolfe(phi_func, phi_prime, alpha_init=1.0)
                    return res['alpha']
                    
                elif metodo_passo == "Newton":
                    alpha_sym = sp.Symbol('α', real=True)
                    subs_dict = {var: x_k[i] + alpha_sym * d_k[i] for i, var in enumerate(self.symbols)}
                    phi_sym_atual = self.f_sym.subs(subs_dict)
                    res = newton(phi_sym_atual, alpha0=0.1)
                    return res['alpha']
                    
                return 0.1
                
            tol = 10 ** self.tol_input.value()
            max_iter = self.max_iter_input.value()
            
            if self.nome_metodo == "BFGS":
                resultado = bfgs(f_eval, grad_eval, x0, calcular_alpha, tol, max_iter)
            else:
                resultado = fletcher_reeves(f_eval, grad_eval, x0, calcular_alpha, tol, max_iter)
                
            self.results_text.append(f"Método {self.nome_metodo} concluído")
            self.results_text.append(f"Convergência: {'Sim' if resultado['convergiu'] else 'Não (Estourou Iterações)'}")
            self.results_text.append(f"Iterações: {resultado['iteracoes']}")
            otimo = resultado['x_otimo']
            ponto_formatado = ", ".join([f"{v:.6f}" for v in otimo])
            self.results_text.append(f"Ponto Ótimo: ({ponto_formatado})\n")
            
            hist = resultado['historico']
            self.results_text.append("Caminho percorrido:")
            for i, p in enumerate(hist):
                p_fmt = ", ".join([f"{v:.4f}" for v in p])
                self.results_text.append(f"k={i}: ({p_fmt})")
                
            if self.check_gif.isChecked():
                self.gerar_e_mostrar_gif(hist, f_num, self.symbols)
            
        except Exception as e:
            self.results_text.setText(f"Erro: {str(e)}")

    def gerar_e_mostrar_gif(self, historico, f_num, symbols):
        self.results_text.append("\nGerando GIF com análise de convergência... Aguarde.")
        QApplication.processEvents()
        
        num_vars = len(symbols)
        historico_np = np.array(historico)
        caminho_temp = "temp_animacao.gif"
        
        valores_f = [f_num(*p) for p in historico_np]
        
        # --- CORREÇÃO 1: Ponto inicial já é o ótimo (1 frame apenas) ---
        if len(historico_np) == 1:
            historico_np = np.vstack([historico_np, historico_np])
            valores_f.append(valores_f[0])
            
        fig = plt.figure(figsize=(12, 5))
        
        # === GRÁFICO DA DIREITA: Decaimento ===
        ax2 = fig.add_subplot(122)
        ax2.set_title("Decaimento da Função Objetivo")
        ax2.set_xlabel("Iteração (k)")
        ax2.set_ylabel("f(x)")
        
        ax2.set_xlim(0, max(1, len(historico_np) - 1))
        margem_y = (max(valores_f) - min(valores_f)) * 0.1 if max(valores_f) != min(valores_f) else 1.0
        ax2.set_ylim(min(valores_f) - margem_y, max(valores_f) + margem_y)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        linha_decai, = ax2.plot([], [], 'b-', lw=2, marker='o', markersize=4)

        # === GRÁFICO DA ESQUERDA: Trajetória 2D ou 3D ===
        if num_vars == 2:
            ax1 = fig.add_subplot(121)
            x_min, x_max = np.min(historico_np[:, 0]) - 1, np.max(historico_np[:, 0]) + 1
            y_min, y_max = np.min(historico_np[:, 1]) - 1, np.max(historico_np[:, 1]) + 1
            X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            Z = f_num(X, Y)
            
            ax1.contour(X, Y, Z, levels=30, cmap='viridis')
            ax1.set_xlabel(str(symbols[0]))
            ax1.set_ylabel(str(symbols[1]))
            
            # --- CORREÇÃO 2: Forçando a aparição dos eixos no Matplotlib ---
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            linha_traj, = ax1.plot([], [], 'r-', lw=2)
            ponto_traj, = ax1.plot([], [], 'bo', markersize=8)
            
            def atualizar_2d(frame):
                linha_traj.set_data(historico_np[:frame+1, 0], historico_np[:frame+1, 1])
                ponto_traj.set_data([historico_np[frame, 0]], [historico_np[frame, 1]])
                ax1.set_title(f"Trajetória 2D - Iteração {frame}")
                linha_decai.set_data(range(frame+1), valores_f[:frame+1])
                return linha_traj, ponto_traj, linha_decai
                
            # --- CORREÇÃO 3: blit=False (resolve os números invisíveis) ---
            anim = FuncAnimation(fig, atualizar_2d, frames=len(historico_np), interval=200, blit=False)
            
        elif num_vars == 3:
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_xlim(np.min(historico_np[:, 0]) - 1, np.max(historico_np[:, 0]) + 1)
            ax1.set_ylim(np.min(historico_np[:, 1]) - 1, np.max(historico_np[:, 1]) + 1)
            ax1.set_zlim(np.min(historico_np[:, 2]) - 1, np.max(historico_np[:, 2]) + 1)
            ax1.set_xlabel(str(symbols[0]))
            ax1.set_ylabel(str(symbols[1]))
            ax1.set_zlabel(str(symbols[2]))
            
            linha_traj, = ax1.plot([], [], [], 'r-', lw=2)
            ponto_traj, = ax1.plot([], [], [], 'bo', markersize=8)
            
            def atualizar_3d(frame):
                linha_traj.set_data(historico_np[:frame+1, 0], historico_np[:frame+1, 1])
                linha_traj.set_3d_properties(historico_np[:frame+1, 2])
                ponto_traj.set_data([historico_np[frame, 0]], [historico_np[frame, 1]])
                ponto_traj.set_3d_properties([historico_np[frame, 2]])
                ax1.set_title(f"Trajetória 3D - Iteração {frame}")
                linha_decai.set_data(range(frame+1), valores_f[:frame+1])
                return linha_traj, ponto_traj, linha_decai
                
            anim = FuncAnimation(fig, atualizar_3d, frames=len(historico_np), interval=200, blit=False)
            
        else:
            self.results_text.append("Suportado apenas 2 ou 3 variáveis para GIF.")
            return

        # --- CORREÇÃO 4: Margens manuais para impedir que os textos vazem do GIF ---
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.3)
        
        escritor = PillowWriter(fps=5)
        anim.save(caminho_temp, writer=escritor)
        plt.close(fig)
        
        self.results_text.append("GIF gerado")
        self.janela_gif = JanelaGif(caminho_temp, self)
        self.janela_gif.exec_()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Otimização Multidimensional")
        self.setGeometry(100, 100, 500, 600)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(ModuloOtimizacaoBase("Fletcher-Reeves"), "Fletcher-Reeves")
        self.tabs.addTab(ModuloOtimizacaoBase("BFGS"), "BFGS")
        self.setCentralWidget(self.tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())