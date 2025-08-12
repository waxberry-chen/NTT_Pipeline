#Clock signal
create_clock -period 41.666 -name sys_clk_pin -add [get_ports lbus_clkn]



#================================================ Pin assignment
#------------------------------------------------ Clock, reset, LED, and SW.
#################
# CLOCK / RESET #
#################
#NET "osc_en_b" LOC="J8" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
set_property PACKAGE_PIN J8      [get_ports {osc_en_b}]
set_property IOSTANDARD LVCMOS25 [get_ports {osc_en_b}]
set_property SLEW FAST           [get_ports {osc_en_b}]
set_property DRIVE 12            [get_ports {osc_en_b}]



#######
# LED #
#######
# NET "led<9>" LOC="G20" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
# NET "led<8>" LOC="L19" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
# NET "led<7>" LOC="K18" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
# NET "led<6>" LOC="H19" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
# NET "led<5>" LOC="K15" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
# NET "led<4>" LOC="P16" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
# NET "led<3>" LOC="T19" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
# NET "led<2>" LOC="T18" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
# NET "led<1>" LOC="H12" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
# NET "led<0>" LOC="H11" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;


set_property PACKAGE_PIN G20 [get_ports {led[9]}]
set_property PACKAGE_PIN L19 [get_ports {led[8]}]
set_property PACKAGE_PIN K18 [get_ports {led[7]}]
set_property PACKAGE_PIN H19 [get_ports {led[6]}]
set_property PACKAGE_PIN K15 [get_ports {led[5]}]
set_property PACKAGE_PIN P16 [get_ports {led[4]}]
set_property PACKAGE_PIN T19 [get_ports {led[3]}]
set_property PACKAGE_PIN T18 [get_ports {led[2]}]
set_property PACKAGE_PIN H12 [get_ports {led[1]}]
set_property PACKAGE_PIN H11 [get_ports {led[0]}]


set_property IOSTANDARD LVCMOS25 [get_ports {led[9]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[8]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[7]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[6]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[5]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[4]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[1]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[0]}]

set_property SLEW FAST [get_ports {led[9]}]
set_property SLEW FAST [get_ports {led[8]}]
set_property SLEW FAST [get_ports {led[7]}]
set_property SLEW FAST [get_ports {led[6]}]
set_property SLEW FAST [get_ports {led[5]}]
set_property SLEW FAST [get_ports {led[4]}]
set_property SLEW FAST [get_ports {led[3]}]
set_property SLEW FAST [get_ports {led[2]}]
set_property SLEW FAST [get_ports {led[1]}]
set_property SLEW FAST [get_ports {led[0]}]


set_property DRIVE 12 [get_ports {led[9]}]
set_property DRIVE 12 [get_ports {led[8]}]
set_property DRIVE 12 [get_ports {led[7]}]
set_property DRIVE 12 [get_ports {led[6]}]
set_property DRIVE 12 [get_ports {led[5]}]
set_property DRIVE 12 [get_ports {led[4]}]
set_property DRIVE 12 [get_ports {led[3]}]
set_property DRIVE 12 [get_ports {led[2]}]
set_property DRIVE 12 [get_ports {led[1]}]
set_property DRIVE 12 [get_ports {led[0]}]

########
# GPIO #
########
# NET "gpio_startn" LOC="D19" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
# NET "gpio_endn"   LOC="N17" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;
# NET "gpio_exec"   LOC="N16" |IOSTANDARD=LVCMOS25 |SLEW=QUIETIO |DRIVE=2 |TIG;


set_property PACKAGE_PIN D19     [get_ports {gpio_startn}]
set_property IOSTANDARD LVCMOS25 [get_ports {gpio_startn}]
set_property SLEW FAST           [get_ports {gpio_startn}]
set_property DRIVE 12            [get_ports {gpio_startn}]

set_property PACKAGE_PIN N17     [get_ports {gpio_endn}]
set_property IOSTANDARD LVCMOS25 [get_ports {gpio_endn}]
set_property SLEW FAST           [get_ports {gpio_endn}]
set_property DRIVE 12            [get_ports {gpio_endn}]

set_property PACKAGE_PIN N16     [get_ports {gpio_exec}]
set_property IOSTANDARD LVCMOS25 [get_ports {gpio_exec}]
set_property SLEW FAST           [get_ports {gpio_exec}]
set_property DRIVE 12            [get_ports {gpio_exec}]


#------------------------------------------------ Local bus
#############################################
# Spartan-6 HPIC (LVCMOS15, SSTL15 or HTSL) #
#############################################
# NET "lbus_clkn"   LOC="AB11" |IOSTANDARD=LVCMOS15;
# NET "lbus_rstn"   LOC="AA13" |IOSTANDARD=LVCMOS15;

# NET "lbus_do<0>"  LOC="V4"   |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<1>"  LOC="V2"   |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<2>"  LOC="W1"   |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<3>"  LOC="AB1"  |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<4>"  LOC="Y3"   |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<5>"  LOC="U7"   |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<6>"  LOC="V3"   |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<7>"  LOC="AF10" |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<8>"  LOC="AC13" |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<9>"  LOC="AE12" |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<10>" LOC="U6"   |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<11>" LOC="AE13" |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<12>" LOC="AA10" |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<13>" LOC="AB12" |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<14>" LOC="AA4"  |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_do<15>" LOC="AE8"  |IOSTANDARD=LVCMOS15 |SLEW=QUIETIO |DRIVE=2;
# NET "lbus_wrn"    LOC="AD10" |IOSTANDARD=LVCMOS15;
# NET "lbus_rdn"    LOC="Y13"  |IOSTANDARD=LVCMOS15;


set_property PACKAGE_PIN AB11    [get_ports {lbus_clkn}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_clkn}]
set_property PACKAGE_PIN AA13    [get_ports {lbus_rstn}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_rstn}]


set_property PACKAGE_PIN V4   [get_ports {lbus_do[0]}]
set_property PACKAGE_PIN V2   [get_ports {lbus_do[1]}]
set_property PACKAGE_PIN W1   [get_ports {lbus_do[2]}]
set_property PACKAGE_PIN AB1  [get_ports {lbus_do[3]}]
set_property PACKAGE_PIN Y3   [get_ports {lbus_do[4]}]
set_property PACKAGE_PIN U7   [get_ports {lbus_do[5]}]
set_property PACKAGE_PIN V3   [get_ports {lbus_do[6]}]
set_property PACKAGE_PIN AF10 [get_ports {lbus_do[7]}]
set_property PACKAGE_PIN AC13 [get_ports {lbus_do[8]}]
set_property PACKAGE_PIN AE12 [get_ports {lbus_do[9]}]
set_property PACKAGE_PIN U6   [get_ports {lbus_do[10]}]
set_property PACKAGE_PIN AE13 [get_ports {lbus_do[11]}]
set_property PACKAGE_PIN AA10 [get_ports {lbus_do[12]}]
set_property PACKAGE_PIN AB12 [get_ports {lbus_do[13]}]
set_property PACKAGE_PIN AA4  [get_ports {lbus_do[14]}]
set_property PACKAGE_PIN AE8  [get_ports {lbus_do[15]}]


set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[0]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[1]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[2]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[3]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[4]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[5]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[6]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[7]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[8]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[9]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[10]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[11]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[12]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[13]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[14]}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_do[15]}]

set_property SLEW FAST [get_ports {lbus_do[0]}]
set_property SLEW FAST [get_ports {lbus_do[1]}]
set_property SLEW FAST [get_ports {lbus_do[2]}]
set_property SLEW FAST [get_ports {lbus_do[3]}]
set_property SLEW FAST [get_ports {lbus_do[4]}]
set_property SLEW FAST [get_ports {lbus_do[5]}]
set_property SLEW FAST [get_ports {lbus_do[6]}]
set_property SLEW FAST [get_ports {lbus_do[7]}]
set_property SLEW FAST [get_ports {lbus_do[8]}]
set_property SLEW FAST [get_ports {lbus_do[9]}]
set_property SLEW FAST [get_ports {lbus_do[10]}]
set_property SLEW FAST [get_ports {lbus_do[11]}]
set_property SLEW FAST [get_ports {lbus_do[12]}]
set_property SLEW FAST [get_ports {lbus_do[13]}]
set_property SLEW FAST [get_ports {lbus_do[14]}]
set_property SLEW FAST [get_ports {lbus_do[15]}]


set_property DRIVE 12 [get_ports {lbus_do[9]}]
set_property DRIVE 12 [get_ports {lbus_do[8]}]
set_property DRIVE 12 [get_ports {lbus_do[7]}]
set_property DRIVE 12 [get_ports {lbus_do[6]}]
set_property DRIVE 12 [get_ports {lbus_do[5]}]
set_property DRIVE 12 [get_ports {lbus_do[4]}]
set_property DRIVE 12 [get_ports {lbus_do[3]}]
set_property DRIVE 12 [get_ports {lbus_do[2]}]
set_property DRIVE 12 [get_ports {lbus_do[1]}]
set_property DRIVE 12 [get_ports {lbus_do[0]}]
set_property DRIVE 12 [get_ports {lbus_do[5]}]
set_property DRIVE 12 [get_ports {lbus_do[4]}]
set_property DRIVE 12 [get_ports {lbus_do[3]}]
set_property DRIVE 12 [get_ports {lbus_do[2]}]
set_property DRIVE 12 [get_ports {lbus_do[1]}]
set_property DRIVE 12 [get_ports {lbus_do[0]}]



set_property PACKAGE_PIN AD10    [get_ports {lbus_wrn}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_wrn}]
set_property PACKAGE_PIN Y13     [get_ports {lbus_rdn}]
set_property IOSTANDARD LVCMOS15 [get_ports {lbus_rdn}]

########################################
# Spartan-6 HRIC (LVCMOS25 or LVDS_25) #
########################################
# NET "lbus_di_a<0>"   LOC="T22" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<1>"   LOC="M24" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<2>"   LOC="K25" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<3>"   LOC="R26" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<4>"   LOC="M25" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<5>"   LOC="U17" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<6>"   LOC="N26" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<7>"   LOC="R16" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<8>"   LOC="T20" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<9>"   LOC="R22" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<10>"  LOC="M21" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<11>"  LOC="T24" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<12>"  LOC="P23" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<13>"  LOC="N21" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<14>"  LOC="R21" |IOSTANDARD=LVCMOS25;
# NET "lbus_di_a<15>"  LOC="N18" |IOSTANDARD=LVCMOS25;



set_property PACKAGE_PIN T22  [get_ports {lbus_di_a[0]}]
set_property PACKAGE_PIN M24  [get_ports {lbus_di_a[1]}]
set_property PACKAGE_PIN K25  [get_ports {lbus_di_a[2]}]
set_property PACKAGE_PIN R26  [get_ports {lbus_di_a[3]}]
set_property PACKAGE_PIN M25  [get_ports {lbus_di_a[4]}]
set_property PACKAGE_PIN U17  [get_ports {lbus_di_a[5]}]
set_property PACKAGE_PIN N26  [get_ports {lbus_di_a[6]}]
set_property PACKAGE_PIN R16  [get_ports {lbus_di_a[7]}]
set_property PACKAGE_PIN T20  [get_ports {lbus_di_a[8]}]
set_property PACKAGE_PIN R22  [get_ports {lbus_di_a[9]}]
set_property PACKAGE_PIN M21  [get_ports {lbus_di_a[10]}]
set_property PACKAGE_PIN T24  [get_ports {lbus_di_a[11]}]
set_property PACKAGE_PIN P23  [get_ports {lbus_di_a[12]}]
set_property PACKAGE_PIN N21  [get_ports {lbus_di_a[13]}]
set_property PACKAGE_PIN R21  [get_ports {lbus_di_a[14]}]
set_property PACKAGE_PIN N18  [get_ports {lbus_di_a[15]}]


set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[0]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[1]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[2]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[3]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[4]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[5]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[6]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[7]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[8]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[9]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[10]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[11]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[12]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[13]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[14]}]
set_property IOSTANDARD LVCMOS25 [get_ports {lbus_di_a[15]}]



