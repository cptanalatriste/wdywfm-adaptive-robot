from typing import List, Tuple


class ExperimentRun(object):

    def __init__(self, experiment_name, simulation_id, commands_per_scenario, random_seed, normal_staff_number,
                 staff_number,
                 passenger_number, fall_length):
        # type: (str, int, List[Tuple[str, bool]], int, int, int, int, int) -> None

        self.experiment_name = experiment_name  # type: str
        self.simulation_id = simulation_id  # type: int
        self.base_scenario_commands = commands_per_scenario  # type: List[Tuple[str, bool]]

        self.random_seed = random_seed  # type: int
        self.normal_staff_number = normal_staff_number  # type: int
        self.staff_number = staff_number  # type: int
        self.passenger_number = passenger_number  # type: int
        self.fall_length = fall_length  # type: int

        self.evacuation_time = -1.0  # type: float

    def get_random_seed_command(self):
        # type: () -> str
        return SEED_SIMULATION_REPORTER.format(self.random_seed)

    def get_pre_setup_commands(self):
        # type: () -> List[str]
        pre_setup_commands = [command for command, before_setup in self.base_scenario_commands
                              if before_setup]  # type: List[str]

        pre_setup_commands.append(SET_NUMBER_NORMAL_STAFF_COMMAND.format(self.normal_staff_number))
        pre_setup_commands.append(SET_NUMBER_STAFF_COMMAND.format(self.staff_number))
        pre_setup_commands.append(SET_NUMBER_PASSENGERS_COMMAND.format(self.passenger_number))

        return pre_setup_commands

    def get_post_setup_commands(self):
        # type: () -> List[str]
        post_setup_commands = [command for command, before_setup in self.base_scenario_commands if
                               not before_setup]  # type: List[str]

        post_setup_commands.append(SET_FALL_LENGTH_COMMAND.format(self.fall_length))
        netlogo_simulation_id = '"{}_id_{}_seed_{}_fall_{}"'.format(self.experiment_name, self.simulation_id,
                                                                    self.random_seed, self.fall_length)  # type: str
        post_setup_commands.append(SET_SIMULATION_ID_COMMAND.format(netlogo_simulation_id))

        return post_setup_commands


NETLOGO_PROJECT_DIRECTORY = "/home/cgc87/github/robot-assisted-evacuation/"  # type:str
NETLOGO_MODEL_FILE = NETLOGO_PROJECT_DIRECTORY + "/impact2.10.7/v2.11.0.nlogo"  # type:str
NETLOGO_HOME = "/home/cgc87/netlogo-5.3.1-64"  # type:str
NETLOGO_VERSION = "5"  # type:str
TURTLE_PRESENT_REPORTER = "count turtles"  # type:str
EVACUATED_REPORTER = "number_passengers - count agents + 1"  # type:str
DEAD_REPORTER = "count agents with [ st_dead = 1 ]"  # type:str
SEED_SIMULATION_REPORTER = "seed-simulation {}"
SET_SIMULATION_ID_COMMAND = "set SIMULATION_ID {}"  # type:str
SET_STAFF_SUPPORT_COMMAND = "set REQUEST_STAFF_SUPPORT {}"  # type: str
SET_PASSENGER_SUPPORT_COMMAND = "set REQUEST_BYSTANDER_SUPPORT {}"  # type: str
SET_FALL_LENGTH_COMMAND = "set DEFAULT_FALL_LENGTH {}"  # type:str
SET_ENABLE_LOGGING_COMMAND = "set ENABLE_LOGGING {}"  # type:str
SET_GENERATE_FRAMES_COMMAND = "set ENABLE_FRAME_GENERATION {}"  # type:str
SET_NUMBER_PASSENGERS_COMMAND = "set number_passengers {}"  # type:str
SET_NUMBER_NORMAL_STAFF_COMMAND = "set _number_normal_staff_members {}"  # type:str
SET_NUMBER_STAFF_COMMAND = "set _number_staff_members {}"  # type:str
ENABLE_STAFF_COMMAND = SET_STAFF_SUPPORT_COMMAND.format("TRUE")  # type:str
ENABLE_PASSENGER_COMMAND = SET_PASSENGER_SUPPORT_COMMAND.format("TRUE")  # type:str
NETLOGO_MINIMUM_SEED = -2147483648  # type:int
NETLOGO_MAXIMUM_SEED = 2147483647  # type:int
MAX_NETLOGO_TICKS = 2000  # type: int
