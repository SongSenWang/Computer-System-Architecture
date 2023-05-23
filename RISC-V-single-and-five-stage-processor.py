import os
import argparse


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

class State(object):
    def __init__(self):
        self.IF = {"nop": bool(False), "PC": int(0), "taken": bool(False)}
        self.ID = {"nop": bool(False), "instr": str("0"*32), "PC": int(0), "hazard_nop": bool(False)}
        self.EX = {"nop": bool(False), "instr": str("0"*32), "Read_data1": str("0"*32), "Read_data2": str("0"*32), "Imm": str("0"*32), "Rs": str("0"*5), "Rt": str("0"*5), "Wrt_reg_addr": str("0"*5), "is_I_type": bool(False), "rd_mem": bool(False), 
                   "wrt_mem": bool(False), "alu_op": str("00"), "wrt_enable": bool(False)} # alu_op 00 -> add, 01 -> and, 10 -> or, 11 -> xor
        self.MEM = {"nop": bool(False), "ALUresult": str("0"*32), "Store_data": str("0"*32), "Rs": str("0"*5), "Rt": str("0"*5), "Wrt_reg_addr": str("0"*5), "rd_mem": bool(False), 
                   "wrt_mem": bool(False), "wrt_enable": bool(False)}
        self.WB = {"nop": bool(False), "Wrt_data": str("0"*32), "Rs": str("0"*5), "Rt": str("0"*5), "Wrt_reg_addr": str("0"*5), "wrt_enable": bool(False)}


class Core(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.inst = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem



#------------------------------------
# single cycle functions

# ALU arithmetic implement
def Calculate_R(funct7, funct3, rs1, rs2):
    rd = 0
    # ADD
    if funct7 == 0b0000000 and funct3 == 0b000:
        rd = rs1 + rs2

    # SUB
    if funct7 == 0b0100000 and funct3 == 0b000:
        rd = rs1 - rs2

    # XOR
    if funct7 == 0b0000000 and funct3 == 0b100:
        rd = rs1 ^ rs2

    # OR
    if funct7 == 0b0000000 and funct3 == 0b110:
        rd = rs1 | rs2

    # AND
    if funct7 == 0b0000000 and funct3 == 0b111:
        rd = rs1 & rs2

    return rd

# compute sign extended immediate, sign bit:most significant bit location
def sign_extend(val, sign_bit):

    if (val & (1 << sign_bit)) != 0:  # get sign bit, if is set 
        val = val - (1 << (sign_bit + 1))  # negative value complement
    return val  

def Calculate_I(funct3, rs1, imm):
    rd = 0
    # ADDI
    if funct3 == 0b000:
        rd = rs1 + sign_extend(imm, 11)

    # XORI
    if funct3 == 0b100:
        rd = rs1 ^ sign_extend(imm, 11)

    # ORI
    if funct3 == 0b110:
        rd = rs1 | sign_extend(imm, 11)

    # ANDI
    if funct3 == 0b111:
        rd = rs1 & sign_extend(imm, 11)

    return rd

    
# single cycle cpu
class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(SingleStageCore, self).__init__(ioDir + os.sep + "SS_", imem, dmem)
        self.opFilePath = ioDir + os.sep + "StateResult_SS.txt"

    def step(self):
        # implementation of each instruction

        fetchedInstr = int(self.ext_imem.readInstr(self.state.IF["PC"]), 16) # hex into integer
        opcode = fetchedInstr & (2 ** 7 - 1) # least significant 7 bits

        # decode and then execute
        self.Decode(opcode, fetchedInstr)
        
        self.halted = False
        if self.state.IF["nop"]:
            self.halted = True
        
        if not self.state.IF["taken"] and self.state.IF["PC"] + 4 < len(self.ext_imem.IMem):
            self.nextState.IF["PC"] = self.state.IF["PC"] + 4
        else:
            self.state.IF["taken"] = False # take branch, then set taken to False again
            
        self.myRF.outputRF(self.cycle) # output file of registers after each cycle
        self.printState(self.nextState, self.cycle) # print states after each cycle
            
        self.state = self.nextState #The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1
        self.inst += 1 # instruction counter

    def Decode(self, opcode, fetchedInstr):
        # R-type
        if opcode == 0b0110011:

            # get funct7
            funct7 = fetchedInstr >> 25
            # get funct3
            funct3 = (fetchedInstr >> 12) & ((1 << 3) - 1)
            # get rs2
            rs2 = (fetchedInstr >> 20) & ((1 << 5) - 1)
            # get rs1
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            # get rd
            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            # get data in rs1
            data_rs1 = self.myRF.readRF(rs1)
            # get data in rs2
            data_rs2 = self.myRF.readRF(rs2)
            # get result data
            data_rd = Calculate_R(funct7, funct3, data_rs1, data_rs2)
            # store all fetched and computed data
            self.myRF.writeRF(rd, data_rd)

        # I Type
        elif opcode == 0b0010011:

            # get immediate
            imm = fetchedInstr >> 20 & ((1 << 12) - 1)

            # get funct3
            funct3 = (fetchedInstr >> 12) & ((1 << 3) - 1)
            # get rs1
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            # get rd
            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            # get data in rs1
            data_rs1 = self.myRF.readRF(rs1)
            # get result data
            data_rd = Calculate_I(funct3, data_rs1, imm)
            # store result data in rd register
            self.myRF.writeRF(rd, data_rd)

        # J Type Jal
        elif opcode == 0b1101111:

            # get imm
            imm19_12 = (fetchedInstr >> 12) & ((1 << 8) - 1)
            imm11 = (fetchedInstr >> 20) & 1
            imm10_1 = (fetchedInstr >> 21) & ((1 << 10) - 1)
            imm20 = (fetchedInstr >> 31) & 1
            imm = (imm20 << 20) | (imm10_1 << 1) | (imm11 << 11) | (imm19_12 << 12)

            # get rd
            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            self.myRF.writeRF(rd, self.state.IF["PC"] + 4)
            self.nextState.IF["PC"] = self.state.IF["PC"] + sign_extend(imm, 20)
            self.state.IF["taken"] = True

        # B Type
        elif opcode == 0b1100011:

            # get imm
            imm11 = (fetchedInstr >> 7) & 1
            imm4_1 = (fetchedInstr >> 8) & ((1 << 4) - 1)
            imm10_5 = (fetchedInstr >> 25) & ((1 << 6) - 1)
            imm12 = (fetchedInstr >> 31) & 1
            imm = (imm11 << 11) | (imm4_1 << 1) | (imm10_5 << 5) | (imm12 << 12)

            # get rs2
            rs2 = (fetchedInstr >> 20) & ((1 << 5) - 1)
            # get rs1
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            # get funct3
            funct3 = (fetchedInstr >> 12) & ((1 << 3) - 1)

            # BEQ
            if funct3 == 0b000:
                data_rs1 = self.myRF.readRF(rs1)
                data_rs2 = self.myRF.readRF(rs2)
                if data_rs1 == data_rs2:
                    self.nextState.IF["PC"] = self.state.IF["PC"] + sign_extend(imm, 12)
                    self.state.IF["taken"] = True

            # BNE
            else:
                data_rs1 = self.myRF.readRF(rs1)
                data_rs2 = self.myRF.readRF(rs2)
                if data_rs1 != data_rs2:
                    self.nextState.IF["PC"] = self.state.IF["PC"] + sign_extend(imm, 12)
                    self.state.IF["taken"] = True

        # LW
        elif opcode == 0b0000011:

            # get imm
            imm = fetchedInstr >> 20
            # get rs1
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            # get rd
            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            self.myRF.writeRF(Reg_addr=rd,
                              Wrt_reg_data=int(self.ext_dmem.readDataMem(
                                  ReadAddress=self.myRF.readRF(rs1) + sign_extend(imm, 11)), 16))

        # SW
        elif opcode == 0b0100011:

            # get imm
            imm11_5 = fetchedInstr >> 25
            imm4_0 = (fetchedInstr >> 7) & ((1 << 5) - 1)
            imm = (imm11_5 << 5) | imm4_0

            # get funct3
            funct3 = fetchedInstr & (((1 << 3) - 1) << 12)
            # get rs1
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            # get rd
            rs2 = (fetchedInstr >> 20) & ((1 << 5) - 1)

            self.ext_dmem.writeDataMem(Address=(rs1 + sign_extend(imm, 11)) & ((1 << 32) - 1),
                                       WriteData=self.myRF.readRF(rs2))

        # HALT
        else:
            self.state.IF["nop"] = True

    # print StateResult_SS.txt
    def printState(self, state, cycle):
        printstate = ["State after executing cycle: " + str(cycle) + "\n"] # "-"*70+"\n",    dividing line
        printstate.append("IF.PC: " + str(state.IF["PC"]) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n")
        
        if(cycle == 0): 
            perm = "w"
        else: 
            perm = "a"

        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)


#-----------------------------------------
# five stages
class InstructionFetchState:
    def __init__(self) -> None:
        self.nop: bool = False
        self.PC: int = 0

    def __dict__(self):
        return {"PC": self.PC, "nop": self.nop}

class InstructionDecodeState:
    def __init__(self) -> None:
        self.nop: bool = True
        self.hazard_nop: bool = False
        self.PC: int = 0
        self.instr: str = "0"*32

    def __dict__(self):
        return {"Instr": self.instr[::-1], "nop": self.nop}

class ExecutionState:
    def __init__(self) -> None:
        self.nop: bool = True
        self.instr: str = ""
        self.read_data_1: str = "0" * 32
        self.read_data_2: str = "0" * 32
        self.imm: str = "0" * 32
        self.rs: str = "0" * 5
        self.rt: str = "0" * 5
        self.write_reg_addr: str = "0" * 5
        self.is_I_type: bool = False
        self.read_mem: bool = False
        self.write_mem: bool = False
        self.alu_op: str = "00" # 00 -> add, 01 -> and, 10 -> or, 11 -> xor
        self.write_enable: bool = False

    def __dict__(self):
        return {
            "nop": self.nop,
            "instr": self.instr[::-1],
            "Operand1": self.read_data_1,
            "Operand2": self.read_data_2,
            "Imm": self.imm,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "is_I_type": int(self.is_I_type),
            "rd_mem": int(self.read_mem),
            "wrt_mem": int(self.write_mem),
            "alu_op": "".join(list(map(str, self.alu_op))),
            "wrt_enable": int(self.write_enable),
        }

class MemoryAccessState:
    def __init__(self) -> None:
        self.nop: bool = True
        self.alu_result: str = "0" * 32
        self.store_data: str = "0" * 32
        self.rs: str = "0" * 5
        self.rt: str = "0" * 5
        self.write_reg_addr: str = "0" * 5
        self.read_mem: bool = False
        self.write_mem: bool = False
        self.write_enable: bool = False

    def __dict__(self):
        return {
            "nop": self.nop,
            "ALUresult": self.alu_result,
            "Store_data": self.store_data,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "rd_mem": int(self.read_mem),
            "wrt_mem": int(self.write_mem),
            "wrt_enable": int(self.write_enable),
        }

class WriteBackState:
    def __init__(self) -> None:
        self.nop: bool = True
        self.write_data: str = "0" * 32
        self.rs: str = "0" * 5
        self.rt: str = "0" * 5
        self.write_reg_addr: str = "0" * 5
        self.write_enable: bool = False

    def __dict__(self):
        return {
            "nop": self.nop,
            "Wrt_data": self.write_data,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "wrt_enable": int(self.write_enable),
        }

class State_five(object):
    def __init__(self):
        self.IF = InstructionFetchState()
        self.ID = InstructionDecodeState()
        self.EX = ExecutionState()
        self.MEM = MemoryAccessState()
        self.WB = WriteBackState()

    def next(self):
        self.ID = InstructionDecodeState()
        self.EX = ExecutionState()
        self.MEM = MemoryAccessState()
        self.WB = WriteBackState()

class Core_five(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.num_instr = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State_five()
        self.nextState = State_five()
        self.ext_imem = imem
        self.ext_dmem = dmem

def int2bin(x: int, n_bits: int = 32) -> str:
    bin_x = bin(x & (2**n_bits - 1))[2:]
    return "0" * (n_bits - len(bin_x)) + bin_x


def bin2int(x: str, sign_ext: bool = False) -> int:
    x = str(x)
    if sign_ext and x[0] == "1":
        return -(-int(x, 2) & (2 ** len(x) - 1))
    return int(x, 2)


class InstructionFetchStage:
    def __init__(
        self,
        state: State_five,
        ins_mem: InsMem,
    ):
        self.state = state
        self.ins_mem = ins_mem

    def run(self):
        if self.state.IF.nop or self.state.ID.nop or (self.state.ID.hazard_nop and self.state.EX.nop):
            return
        instr = self.ins_mem.read_instr(self.state.IF.PC)[::-1]
        if instr == "1" * 32:
            self.state.IF.nop = True
            self.state.ID.nop = True
        else:
            self.state.ID.PC = self.state.IF.PC
            self.state.IF.PC += 4
            self.state.ID.instr = instr

class InstructionDecodeStage:
    def __init__(
        self,
        state: State_five,
        rf: RegisterFile,
    ):
        self.state = state
        self.rf = rf

    def detect_hazard(self, rs):
        if rs == self.state.MEM.write_reg_addr and self.state.MEM.read_mem == 0:
            # EX to 1st
            return 2
        elif rs == self.state.WB.write_reg_addr and self.state.WB.write_enable:
            # EX to 2nd
            # MEM to 2nd
            return 1
        elif rs == self.state.MEM.write_reg_addr and self.state.MEM.read_mem != 0:
            # MEM to 1st
            self.state.ID.hazard_nop = True
            return 1
        else:
            return 0

    def read_data(self, rs, forward_signal):
        if forward_signal == 1:
            return self.state.WB.write_data
        elif forward_signal == 2:
            return self.state.MEM.alu_result
        else:
            return self.rf.read_RF(rs)

    def run(self):
        if self.state.ID.nop:
            if not self.state.IF.nop:
                self.state.ID.nop = False
            return

        self.state.EX.instr = self.state.ID.instr
        self.state.EX.is_I_type = False
        self.state.EX.read_mem = False
        self.state.EX.write_mem = False
        self.state.EX.write_enable = False
        self.state.ID.hazard_nop = False
        self.state.EX.write_reg_addr = "000000"

        opcode = self.state.ID.instr[:7][::-1]
        func3 = self.state.ID.instr[12:15][::-1]

        if opcode == "0110011":
            # r-type instruction
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]

            forward_signal_1 = self.detect_hazard(rs1)
            forward_signal_2 = self.detect_hazard(rs2)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)
            self.state.EX.read_data_2 = self.read_data(rs2, forward_signal_2)

            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.write_enable = True

            func7 = self.state.ID.instr[25:][::-1]

            if func3 == "000":
                # add and sub instruction
                self.state.EX.alu_op = "00"
                if func7 == "0100000":
                    self.state.EX.read_data_2 = int2bin(
                        -bin2int(self.state.EX.read_data_2, sign_ext=True)
                    )
            elif func3 == "111":
                # and instruction
                self.state.EX.alu_op = "01"
            elif func3 == "110":
                # or instruction
                self.state.EX.alu_op = "10"
            elif func3 == "100":
                # xor instruction
                self.state.EX.alu_op = "11"

        elif opcode == "0010011" or opcode == "0000011":
            # i-type instruction
            rs1 = self.state.ID.instr[15:20][::-1]

            forward_signal_1 = self.detect_hazard(rs1)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)

            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.is_I_type = True

            self.state.EX.imm = self.state.ID.instr[20:][::-1]
            self.state.EX.write_enable = True
            self.state.EX.read_mem = opcode == "0000011"

            if func3 == "000":
                # add instruction
                self.state.EX.alu_op = "00"
            elif func3 == "111":
                # and instruction
                self.state.EX.alu_op = "01"
            elif func3 == "110":
                # or instruction
                self.state.EX.alu_op = "10"
            elif func3 == "100":
                # xor instruction
                self.state.EX.alu_op = "11"
        elif opcode == "1101111":
            # j-type instruction
            self.state.EX.imm = (
                "0"
                + self.state.ID.instr[21:31]
                + self.state.ID.instr[20]
                + self.state.ID.instr[12:20]
                + self.state.ID.instr[31]
            )[::-1]
            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.read_data_1 = int2bin(self.state.ID.PC)
            self.state.EX.read_data_2 = int2bin(4)
            self.state.EX.write_enable = True
            self.state.EX.alu_op = "00"

            self.state.IF.PC = self.state.ID.PC + bin2int(self.state.EX.imm, sign_ext=True)
            self.state.ID.nop = True

        elif opcode == "1100011":
            # b-type instruction
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]

            forward_signal_1 = self.detect_hazard(rs1)
            forward_signal_2 = self.detect_hazard(rs2)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)
            self.state.EX.read_data_2 = self.read_data(rs2, forward_signal_2)
            diff = bin2int(self.state.EX.read_data_1, sign_ext=True) - bin2int(
                self.state.EX.read_data_2, sign_ext=True
            )

            self.state.EX.imm = (
                "0"
                + self.state.ID.instr[8:12]
                + self.state.ID.instr[25:31]
                + self.state.ID.instr[7]
                + self.state.ID.instr[31]
            )[::-1]

            if (diff == 0 and func3 == "000") or (diff != 0 and func3 == "001"):
                self.state.IF.PC = self.state.ID.PC + bin2int(self.state.EX.imm, sign_ext=True)
                self.state.ID.nop = True
                self.state.EX.nop = True
            else:
                self.state.EX.nop = True

        elif opcode == "0100011":
            # sw-type instruction
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]

            forward_signal_1 = self.detect_hazard(rs1)
            forward_signal_2 = self.detect_hazard(rs2)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)
            self.state.EX.read_data_2 = self.read_data(rs2, forward_signal_2)

            self.state.EX.imm = (self.state.ID.instr[7:12] + self.state.ID.instr[25:])[::-1]
            self.state.EX.is_I_type = True
            self.state.EX.write_mem = True
            self.state.EX.alu_op = "00"

        if self.state.IF.nop:
            self.state.ID.nop = True
        return 1

class ExecutionStage:
    def __init__(
        self, 
        state: State_five
    ):
        self.state = state

    def run(self):
        if self.state.EX.nop:
            if not self.state.ID.nop:
                self.state.EX.nop = False
            return

        operand_1 = self.state.EX.read_data_1
        operand_2 = (
            self.state.EX.read_data_2
            if not self.state.EX.is_I_type and not self.state.EX.write_mem
            else self.state.EX.imm
        )

        # ADD
        if self.state.EX.alu_op == "00":
            self.state.MEM.alu_result = int2bin(
                bin2int(operand_1, sign_ext=True) + bin2int(operand_2, sign_ext=True)
            )
        # AND
        elif self.state.EX.alu_op == "01":
            self.state.MEM.alu_result = int2bin(
                bin2int(operand_1, sign_ext=True) & bin2int(operand_2, sign_ext=True)
            )
        # OR
        elif self.state.EX.alu_op == "10":
            self.state.MEM.alu_result = int2bin(
                bin2int(operand_1, sign_ext=True) | bin2int(operand_2, sign_ext=True)
            )
        # XOR
        elif self.state.EX.alu_op == "11":
            self.state.MEM.alu_result = int2bin(
                bin2int(operand_1, sign_ext=True) ^ bin2int(operand_2, sign_ext=True)
            )

        self.state.MEM.rs = self.state.EX.rs
        self.state.MEM.rt = self.state.EX.rt
        self.state.MEM.read_mem = self.state.EX.read_mem
        self.state.MEM.write_mem = self.state.EX.write_mem
        if self.state.EX.write_mem:
            self.state.MEM.store_data = self.state.EX.read_data_2
        self.state.MEM.write_enable = self.state.EX.write_enable
        self.state.MEM.write_reg_addr = self.state.EX.write_reg_addr

        if self.state.ID.nop:
            self.state.EX.nop = True

class MemoryAccessStage:
    def __init__(
        self, 
        state: State_five, 
        data_mem: DataMem
    ):
        self.state = state
        self.data_mem = data_mem

    def run(self):
        if self.state.MEM.nop:
            if not self.state.EX.nop:
                self.state.MEM.nop = False
            return
            
        if self.state.MEM.read_mem != 0:
            self.state.WB.write_data = self.data_mem.read_data_mem(self.state.MEM.alu_result)
        elif self.state.MEM.write_mem != 0:
            self.data_mem.write_data_mem(
                self.state.MEM.alu_result, self.state.MEM.store_data
            )
        else:
            self.state.WB.write_data = self.state.MEM.alu_result
            self.state.MEM.store_data = self.state.MEM.alu_result
        self.state.WB.write_enable = self.state.MEM.write_enable
        self.state.WB.write_reg_addr = self.state.MEM.write_reg_addr

        if self.state.EX.nop:
            self.state.MEM.nop = True

class WriteBackStage:
    def __init__(
        self,
        state: State_five,
        rf: RegisterFile,
    ):
        self.state = state
        self.rf = rf

    def run(self):
        if self.state.WB.nop:
            if not self.state.MEM.nop:
                self.state.WB.nop = False
            return
        if self.state.WB.write_enable:
            self.rf.write_RF(self.state.WB.write_reg_addr, self.state.WB.write_data)

        if self.state.MEM.nop:
            self.state.WB.nop = True


class FiveStageCore(Core_five):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(ioDir + os.sep + "FS_", imem, dmem)
        self.opFilePath = ioDir + os.sep + "StateResult_FS.txt"

        self.if_stage = InstructionFetchStage(self.state, self.ext_imem)
        self.id_stage = InstructionDecodeStage(self.state, self.myRF)
        self.ex_stage = ExecutionStage(self.state)
        self.mem_stage = MemoryAccessStage(self.state, self.ext_dmem)
        self.wb_stage = WriteBackStage(self.state, self.myRF)

        

    def step(self):
        # Your implementation

        if (
            self.state.IF.nop
            and self.state.ID.nop
            and self.state.EX.nop
            and self.state.MEM.nop
            and self.state.WB.nop
        ):
            self.halted = True
        current_instr = self.state.ID.instr
        # --------------------- WB stage ---------------------
        self.wb_stage.run()

        # --------------------- MEM stage --------------------
        self.mem_stage.run()

        # --------------------- EX stage ---------------------
        self.ex_stage.run()

        # --------------------- ID stage ---------------------
        self.id_stage.run()

        # --------------------- IF stage ---------------------
        self.if_stage.run()

        self.myRF.output_RF(self.cycle)  # dump RF
        self.printState(
            self.state, self.cycle
        )  # print states after executing cycle 0, cycle 1, cycle 2 ...

        # self.state.next()  # The end of the cycle and updates the current state with the values calculated in this cycle
        self.num_instr += int(current_instr != self.state.ID.instr)
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]  # "-"*70+"\n",  dividing line
        '''
        # IF
        printstate.append("IF.PC: " + str(state.IF["PC"]) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n" + "\n")
        # ID
        printstate.append("ID.Instr: " + str(format(state.ID["Instr"],'032b')) + "\n")
        printstate.append("ID.nop: " + str(state.ID["nop"]) + "\n" + "\n")
        # EX
        printstate.append("EX.Operand1: " + str(format(state.EX["rs1"],'032b')) + "\n")
        printstate.append("EX.Operand2: " + str(format(state.EX["rs2"],'032b')) + "\n")
        printstate.append("EX.StData: " + str(state.EX["Imm"]) + "\n")  # StData ?
        printstate.append("EX.DestReg: " + str(format(state.EX["rd"],'05b')) + "\n") # 5 bit binary
        printstate.append("EX.AluOperation: " + str(format(state.EX["alu_op"],'02b')) + "\n") # 2 bit binary
        printstate.append("EX.UpdatePC: " + str(state.EX["Imm"]) + "\n")   # need to set another new state.UPC bit
        printstate.append("EX.WBEnable: " + str(state.EX["wrt_enable"]) + "\n")
        printstate.append("EX.RdDMem: " + str(state.EX["rd_mem"]) + "\n")  # "rd_mem" need here
        printstate.append("EX.WrDMem: " + str(state.EX["wrt_mem"]) + "\n") 
        printstate.append("EX.Halt: " + str(state.EX["wrt_mem"]) + "\n")  # different from nop? which is this?
        printstate.append("EX.nop: " + str(state.EX["nop"]) + "\n")
        # MEM
        printstate.append("MEM.ALUresult: " + str(format(state.MEM["ALUresult"],'032b')) + "\n")
        printstate.append("MEM.Store_data: " + str(format(state.MEM["Store_data"],'032b')) + "\n")
        '''

        printstate.append("\n")
        printstate.extend(["IF." + key + ": " + str(val) + "\n" for key, val in state.IF.__dict__().items()])
        printstate.append("\n")
        printstate.extend(["ID." + key + ": " + str(val) + "\n" for key, val in state.ID.__dict__().items()])
        printstate.append("\n")
        printstate.extend(["EX." + key + ": " + str(val) + "\n" for key, val in state.EX.__dict__().items()])
        printstate.append("\n")
        printstate.extend(["MEM." + key + ": " + str(val) + "\n" for key, val in state.MEM.__dict__().items()])
        printstate.append("\n")
        printstate.extend(["WB." + key + ": " + str(val) + "\n" for key, val in state.WB.__dict__().items()])
        
        if(cycle == 0): 
            perm = "w"
        else: 
            perm = "a"
            
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

#-----------------------------------------
# print metrics 
# single cycle metrics:
def single_metrics(opFilePath: str, ss: SingleStageCore):
    ss_metrics = [
        "Single Stage Core Performance Metrics: ",
        f"Number of Cycles taken:  {ss.cycle}",
        f"Cycles per instruction:  {int( (ss.cycle - 1)/ss.inst )}",
        f"Instructions per cycle:  {int( ss.inst/(ssCore.cycle - 1) )}",
    ]

    with open(opFilePath + os.sep + "SingleMetrics.txt", "w") as f:
        f.write("\n".join(ss_metrics))

# five stage metrics:
def five_metrics(opFilePath: str, fs: FiveStageCore):
    # print after add one instr, no need to add one instr
    fs_metrics = [
        "Five Stage Core Performance Metrics:",
        f"Number of Cycles taken:  {fs.cycle}",
        f"Cycles per instruction: {fs.cycle / fs.num_instr}",
        f"Instructions per cycle: {fs.num_instr / fs.cycle}",
    ]

    with open(opFilePath + os.sep + "FiveMetrics.txt", "w") as f:
        f.write("\n".join(fs_metrics))

def Performance_metrics(opFilePath: str, ss: SingleStageCore, fs: FiveStageCore):
    ss_metrics = [
        "Single Stage Core Performance Metrics: ",
        f"Number of Cycles taken:  {ss.cycle}",
        f"Cycles per instruction:  {int( (ss.cycle - 1)/ss.inst )}",
        f"Instructions per cycle:  {int( ss.inst/(ssCore.cycle - 1) )}",
    ]

    
    fs_metrics = [
        "Five Stage Core Performance Metrics:",
        f"Number of Cycles taken:  {fs.cycle}",
        f"Cycles per instruction: {fs.cycle / fs.num_instr}",
        f"Instructions per cycle: {fs.num_instr / fs.cycle}",
    ]

    with open(opFilePath + os.sep + "PerformanceMetrics_Result.txt", "w") as f:
        f.write("\n".join(ss_metrics) + "\n\n" + "\n".join(fs_metrics))

# main  
if __name__ == "__main__":
     
    #parse arguments for input file location

    parser = argparse.ArgumentParser(description='RV32I single and five stage processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()

    # the current directory for code
    ioDir = os.path.abspath(args.iodir)


    # Show the directory for input files
    print("IO Directory:", ioDir)

    # common imem
    imem = InsMem("Imem", ioDir)

    # single stage processor
    dmem_ss = DataMem("SS", ioDir)
    
    ssCore = SingleStageCore(ioDir, imem, dmem_ss) 

    while(True):
        if not ssCore.halted:
            ssCore.step()

        if ssCore.halted:
            ssCore.myRF.outputRF(ssCore.cycle) # output file of registers after last cycle
            ssCore.printState(ssCore.nextState, ssCore.cycle) # print states after last cycle
            ssCore.cycle += 1
            break
    
    # dump SS data mem.
    dmem_ss.outputDataMem()
    
    # five stages processor
    dmem_fs = DataMem("FS", ioDir)

    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    while(True):
        if not fsCore.halted:
            fsCore.step()

        if fsCore.halted:
            break
    
    # dump FS data mem.
    dmem_fs.outputDataMem()

    # print in terminal
    print("Single Stage Core Performance Metrics: ")
    print("Number of Cycles taken: ", ssCore.cycle, end=", ")
    print("Number of Instruction in Imem: ", ssCore.inst, end="\n\n")

    print("Five Stage Core Performance Metrics: ")
    print("Number of Cycles taken: ", fsCore.cycle, end=", ")
    # incrementing num of instructions because of an extra HALT instruction which is never decoded
    fsCore.num_instr += 1
    print("Number of Instruction in Imem: ", fsCore.num_instr , end="\n\n")

    # print in file
    Performance_metrics(ioDir, ssCore, fsCore)

    single_metrics(ioDir, ssCore)
    five_metrics(ioDir, fsCore)
