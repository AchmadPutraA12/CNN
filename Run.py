import subprocess

def run_main_script(num_runs):
    """
    Menjalankan main.py sebanyak num_runs kali.
    
    :param num_runs: Jumlah putaran untuk menjalankan main.py.
    """
    for i in range(num_runs):
        print(f"Menjalankan main.py - Putaran {i + 1} dari {num_runs}")
        
        process = subprocess.run(['python', 'main.py'])
        
        if process.returncode != 0:
            print(f"main.py selesai dengan kode keluaran {process.returncode}")
            
            break
        else:
            print("main.py selesai dengan sukses")

if __name__ == '__main__':
    
    num_runs = 3
    
    run_main_script(num_runs)
