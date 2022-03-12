import numpy as np
import csv


n = 1000
ebno = 6
dim = 9
mid = (dim//2)
h = np.load(f'H6Length{n}.npy')
cnt = 1
with open('./data.csv', mode='w') as csv_file:
    fieldName = []
    fieldName.append('id')

    for i in range(-mid,mid+1) :
        fieldName.append('o'+ str(i))
    fieldName.append('target')

    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldName)

    for it in range(1000):
        s = 2*np.random.randint(2,size=(n,1))-1
        noisePw = 1/(10**(ebno/10))
        noise = np.sqrt(noisePw/2)*np.random.standard_normal((n,1))
        r = np.dot(h,s) + noise
        for i in range(mid,len(r)-mid):
            field = [str(cnt)]
            cnt+=1
            for j in range(dim):
                field.append(str(float(r[i-mid + j])))
            field.append(str(int(1 if s[i]>0 else 0)))
            writer.writerow(field)
        