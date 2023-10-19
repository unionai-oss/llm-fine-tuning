import time

def result_iter():
    words = ["foo", "bar", "baz", "\n", "foo", "abc", "xyz", "\n", "hey"]
    for x in words:
        yield x

result = result_iter()

for w in result:
    print(w, end="", flush=True)
    time.sleep(1)
