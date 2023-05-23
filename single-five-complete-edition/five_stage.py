# five stage processor states, functions

from Memory_Registers import *

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


# five stage cpu
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
