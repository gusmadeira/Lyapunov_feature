# 1. Remove arquivos .txt se eles existirem (silenciosamente)
Remove-Item *.txt -ErrorAction SilentlyContinue

# 2. Copia o arquivo da pasta BASE para a pasta atual
# O PowerShell lida bem com as barras / ou \
Copy-Item "../BASE/teste_lyapunov.py" "./teste_lyapunov.py"

# 3. Executa o Python 
# No Windows, geralmente o comando é 'python', não 'python3'
python teste_lyapunov.py