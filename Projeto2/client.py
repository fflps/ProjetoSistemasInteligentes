import connection as cn
import socket
import numpy as np

def main():
    
    # Inicializa a conexão
    c = cn.connect(2037)
    
    # Verifica se conectou com sucesso
    if isinstance(c, socket.socket):
        
        print("Conectado com sucesso")
        
        # Carrega a tabela Q
        try:
            Q = np.loadtxt('resultado.txt')
        except: 
            Q = np.zeros((6,6))
            