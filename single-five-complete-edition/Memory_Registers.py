# define Imem Dmem and Registers

import os


MemSize = 1000 # memory size, in reality, the memory size should be 2^32, but for this lab, for the space resaon, we keep it as this large number, but the memory is still 32-bit addressable.

class InsMem(object):  # read instruction
    def __init__(self, name, ioDir):
        self.id = name
        
        with open(ioDir + os.sep + "imem.txt") as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def readInstr(self, ReadAddress):
        #read instruction memory
        #return 32 bit hex val
        inst = int("".join(self.IMem[ReadAddress : ReadAddress + 4]),2) # change into decimal number
        return format(inst,'#010x') #'0x'+8 bit hex
    
    def read_instr(self, read_address: int) -> str:
        # read instruction memory
        # return 32 bit str binary instruction
        return "".join(self.IMem[read_address : read_address + 4])
    

""" the following code can also work
    def readInstr(self, ReadAddress):
        inst = 0
        for i in range(ReadAddress, ReadAddress + 4):
            inst = inst | int(self.IMem[i],2) # change into decimal number
            if i < ReadAddress + 3:
                inst = inst<<8
        
        return format(inst,'#010x') #'0x'+8 bit hex
"""
        
          
class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(ioDir + os.sep + "dmem.txt") as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]
        # fill in the empty memory with 0s
        self.DMem = self.DMem + (['00000000'] * (MemSize - len(self.DMem))) 

    def readDataMem(self, ReadAddress):
        #read data memory
        #return 32 bit hex val 8
        data32 = int("".join(self.DMem[ReadAddress : ReadAddress + 4]),2) # change into decimal number
        return format(data32,'#010x') #'0x'+8 bit hex
    
    
    """ the following code can also work
    def readDataMem(self, ReadAddress):        
                data32 = 0
        for i in range(ReadAddress, ReadAddress + 4):
            data32 = data32 | int(self.DMem[i],2) # change into decimal number
            if i != ReadAddress + 3:
                data32 = data32<<8
        
        return format(data32,'#010x') #'0x'+8 bit hex
    """
    
    def writeDataMem(self, Address, WriteData):
        # write data into byte addressable memory
        mask8 = int('0b11111111',2) # 8-bit mask
        data8_arr = []

        for j in range(4):
            data8_arr.append(WriteData & mask8)
            WriteData = WriteData>>8
        
        for i in range(4):
            # most significant bit(last element in data8_arr) in smallest address
            self.DMem[Address + i] = format(data8_arr.pop(),'08b')

    # five stage func
    def read_data_mem(self, read_addr: str) -> str:
        # read data memory
        # return 32 bit hex val
        read_addr_int = bin2int(read_addr)
        return "".join(self.DMem[read_addr_int : read_addr_int + 4])

    def write_data_mem(self, addr: str, write_data: str):
        # write data into byte addressable memory
        addr_int = bin2int(addr)
        for i in range(4):
            self.DMem[addr_int + i] = write_data[8 * i : 8 * (i + 1)]
    
    # output file of Dmem  SS_DMEMResult.txt              
    def outputDataMem(self):
        resPath = self.ioDir + os.sep + self.id + "_DMEMResult.txt"
        with open(resPath, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])

class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        self.Registers = [0x0 for i in range(32)] # 32 registers for single cycle
        self.registers = [int2bin(0) for _ in range(32)] # five stage
    
    def readRF(self, Reg_addr): # read register
        return self.Registers[Reg_addr]
    
    def writeRF(self, Reg_addr, Wrt_reg_data): # write into registers
        if Reg_addr != 0:
            self.Registers[Reg_addr] = Wrt_reg_data & ((1 << 32) - 1) # and 32 bits 1 mask

    # output file of registers  SS_RFResult.txt
    def outputRF(self, cycle):
        op = ["State of RF after executing cycle:  " + str(cycle) + "\n"]   # "-"*70+"\n",  dividing line
        op.extend([format(val,'032b')+"\n" for val in self.Registers])
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)
    
    # five stage
    def read_RF(self, reg_addr: str) -> str:
        # Fill in
        return self.registers[bin2int(reg_addr)]

    def write_RF(self, reg_addr: str, wrt_reg_data: str):
        # Fill in
        if reg_addr == "00000":
            return
        self.registers[bin2int(reg_addr)] = wrt_reg_data

    def output_RF(self, cycle):
        op = ["State of RF after executing cycle:" + str(cycle) + "\n"]
        op.extend([f"{val}" + "\n" for val in self.registers])
        if cycle == 0:
            perm = "w"
        else:
            perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)


def int2bin(x: int, n_bits: int = 32) -> str:
    bin_x = bin(x & (2**n_bits - 1))[2:]
    return "0" * (n_bits - len(bin_x)) + bin_x


def bin2int(x: str, sign_ext: bool = False) -> int:
    x = str(x)
    if sign_ext and x[0] == "1":
        return -(-int(x, 2) & (2 ** len(x) - 1))
    return int(x, 2)