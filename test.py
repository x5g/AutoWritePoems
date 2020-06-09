import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages('modelResults_7.pdf')

plt.figure()

acc = []
loss = []
with open('acc.txt', 'r') as file:
    for line in file.readlines():
        acc.append(float(line))

with open('loss.txt', 'r') as file:
    for line in file.readlines():
        loss.append(float(line))

print(acc)
print(loss)

plt.plot(acc, label='acc')
plt.plot(loss, label='loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('acc & loss')
plt.title('Train results (acc & loss) base on poetry6min.txt')
# plt.show()

plt.tight_layout()

print('savefig...')
pdf.savefig()
plt.close()
pdf.close()

