import time

begin = time.time()
while True:
    print(int(time.time() - begin) % 100)
    time.sleep(1)