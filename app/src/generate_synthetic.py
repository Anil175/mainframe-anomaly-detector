
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

OUT = os.environ.get('DATA_PATH', '/app/data/latest.csv')

def generate(n=500, anomaly_at=450):
    start = datetime.utcnow()
    rows = []
    for i in range(n):
        t = start + timedelta(seconds=i*30)
        cpu = 0.4 + 0.1*np.sin(i/10) + 0.02*np.random.randn()
        io = 100 + 20*np.cos(i/20) + 5*np.random.randn()
        mem = 0.5 + 0.05*np.sin(i/15) + 0.01*np.random.randn()
        jobs = 40 + int(3*np.sin(i/5) + np.random.randint(-1,2))
        if i == anomaly_at:
            cpu += 1.5
            io += 500
        rows.append([t.isoformat(), cpu, io, mem, jobs])
    df = pd.DataFrame(rows, columns=['timestamp','cpu_usage','io_rate','mem_usage','job_count'])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Wrote", OUT)

if __name__ == '__main__':
    generate()
