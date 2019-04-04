import mmap
from struct import *#pack, unpack, Struct
import math

def write_data_to_binary_file(item_list, file_name):
    #print(calcsize('<i f'))
    with open(file_name, "wb") as file_object:
        for item in item_list:
            square = math.sqrt(item)
            file_object.write(
                pack("<i f", item, square))


def get_memory_map_from_binary_file(file_name):
    num_bytes = 25 * 8  #og = 4 ~~~ MODIFY THIS LINE (ii) ~~~

    with open(file_name, "r") as file_object:
        file_map = mmap.mmap(
            file_object.fileno(),
            length=num_bytes,
            access=mmap.ACCESS_READ)
    #print(file_map)
    return num_bytes, file_map


def parse_memory_map(file_map):
    parsed_values = []
    for i in range(25):  # ~~~ MODIFY THIS LINE (iii) ~~~
        parsed_values.append(
            unpack("<i f", file_map[i * 8 : i * 8 + 8]))  # ~~~ MODIFY THIS LINE (iv) ~~~
    return parsed_values


def warmup():
    item_list = range(5,128,5)  # ~~~ MODIFY THIS LINE (v) ~~~

    write_data_to_binary_file(item_list=item_list, file_name="out_warmup.bin")

    num_bytes, file_map = get_memory_map_from_binary_file("out_warmup.bin")

    with open("out_warmup_bytes.txt", "w") as file_object:
        file_object.write(str(num_bytes))

    parsed_values = parse_memory_map(file_map)

    for item in parsed_values:
        print (item)


if __name__ == "__main__":
    warmup()
