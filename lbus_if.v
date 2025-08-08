/*-------------------------------------------------------------------------
 AIST-LSI compatible local bus I/F for AES_Comp on FPGA
 *** NOTE *** 
 This circuit works only with AES_Comp.
 Compatibility for another cipher module may be provided in future release.
 
 File name   : lbus_if.v
 Version     : 1.3
 Created     : APR/02/2012
 Last update : APR/11/2012
 Desgined by : Toshihiro Katashita
 
 
 Copyright (C) 2012 AIST
 
 By using this code, you agree to the following terms and conditions.
 
 This code is copyrighted by AIST ("us").
 
 Permission is hereby granted to copy, reproduce, redistribute or
 otherwise use this code as long as: there is no monetary profit gained
 specifically from the use or reproduction of this code, it is not sold,
 rented, traded or otherwise marketed, and this copyright notice is
 included prominently in any copy made.
 
 We shall not be liable for any damages, including without limitation
 direct, indirect, incidental, special or consequential damages arising
 from the use of this code.
 
 When you publish any results arising from the use of this code, we will
 appreciate it if you can cite our webpage.
(http://www.risec.aist.go.jp/project/sasebo/)
 -------------------------------------------------------------------------*/ 


//================================================ LBUS_IF
module LBUS_IF
  (lbus_a, lbus_di, lbus_do, lbus_wr, lbus_rd, // Local bus
   a,b,
   blk_dout, blk_krdy, blk_drdy, blk_kvld, blk_dvld,
    blk_en, blk_rstn,
   clk, rst);                                  // Clock and reset
   
   //------------------------------------------------
   // Local bus
   input [15:0]   lbus_a;  // Address
   input [15:0]   lbus_di; // Input data  (Controller -> Cryptographic module)
   input          lbus_wr; // Assert input data
   input          lbus_rd; // Assert output data
   output [15:0]  lbus_do; // Output data (Cryptographic module -> Controller)

   // Block cipher
    (* dont_touch = "true" *)(* keep = "TRUE" *) output reg [11:0] a,b;

   input [127:0]  blk_dout;
   output         blk_krdy, blk_drdy;
   input          blk_kvld, blk_dvld;
   output         blk_en;
   output         blk_rstn;

   // Clock and reset
   input         clk, rst;

   //------------------------------------------------
   reg [15:0]    lbus_do;

   reg [256:0]   Key1,  Key2;
   reg [3:0]     Wots_Tree_index;
   reg [6:0]     Wots_Tree_height;
   reg [3:0]     Wots_KeyPair;
   reg [63:0]    Sel_Tree;
   reg [4:0]     Wots_Layer_Addr;

   reg           blk_krdy;
   reg [127:0] 	 blk_dout_reg;
   wire          blk_drdy;
   reg[1:0]      Wots_Mode;
   wire          blk_en = 1;
   reg           blk_rstn;
   
   reg [1:0]     wr;
   reg           trig_wr;
   wire          ctrl_wr;
   reg [2:0]     ctrl;
   reg [3:0]     blk_trig;

   //------------------------------------------------
   always @(posedge clk or posedge rst)
     if (rst) wr <= 2'b00;
     else     wr <= {wr[0],lbus_wr};
   
   always @(posedge clk or posedge rst)
     if (rst)            trig_wr <= 0;
     else if (wr==2'b01) trig_wr <= 1;
     else                trig_wr <= 0;
   
   assign ctrl_wr = (trig_wr & (lbus_a==16'h0002));
   
   always @(posedge clk or posedge rst) 
     if (rst) ctrl <= 3'b000;
     else begin
        if (blk_drdy)       ctrl[0] <= 1;
        else if (|blk_trig) ctrl[0] <= 1;
        else if (blk_dvld)  ctrl[0] <= 0;

        if (blk_krdy)      ctrl[1] <= 1;
        else if (blk_kvld) ctrl[1] <= 0;
        
        ctrl[2] <= ~blk_rstn;
     end

   always @(posedge clk or posedge rst) 
     if (rst)           blk_dout_reg <= 128'h0;
     else if (blk_dvld) blk_dout_reg <= blk_dout;
   
   always @(posedge clk or posedge rst) 
     if (rst)          blk_trig <= 4'h0;
     else if (ctrl_wr) blk_trig <= {lbus_di[0],3'h0};
     else              blk_trig <= {1'h0,blk_trig[3:1]};
   assign blk_drdy = blk_trig[0];

   always @(posedge clk or posedge rst) 
     if (rst)          blk_krdy <= 0;
     else if (ctrl_wr) blk_krdy <= lbus_di[1];
     else              blk_krdy <= 0;

   always @(posedge clk or posedge rst) 
     if (rst)          blk_rstn <= 1;
     else if (ctrl_wr) blk_rstn <= ~lbus_di[2];
     else              blk_rstn <= 1;
   
   //------------------------------------------------
   always @(posedge clk or posedge rst) begin
      if (rst) begin
          a<=12'd0;
          b<=12'd0;
      end else if (trig_wr) begin
        case(lbus_a)
         16'h0100 : a <= lbus_di;
         16'h0102 : b <= lbus_di;
        //  16'h0110 : Key1[143:128] <= lbus_di;
        //  16'h0112 : Key1[159:144] <= lbus_di;
        //  16'h0114 : Key1[175:160] <= lbus_di;
        //  16'h0116 : Key1[191:176] <= lbus_di;
        //  16'h0118 : Key1[207:192] <= lbus_di;
        //  16'h011A : Key1[223:208] <= lbus_di;
        //  16'h011C : Key1[239:224] <= lbus_di;
        //  16'h011E : Key1[255:240] <= lbus_di;


        //  16'h0120 : Key2[ 15:  0] <= lbus_di;
        //  16'h0122 : Key2[ 31: 16] <= lbus_di;
        //  16'h0124 : Key2[ 47: 32] <= lbus_di;
        //  16'h0126 : Key2[ 63: 48] <= lbus_di;
        //  16'h0128 : Key2[ 79: 64] <= lbus_di;
        //  16'h012A : Key2[ 95: 80] <= lbus_di;
        //  16'h012C : Key2[111: 96] <= lbus_di;
        //  16'h012E : Key2[127:112] <= lbus_di;
        //  16'h0130 : Key2[143:128] <= lbus_di;
        //  16'h0132 : Key2[159:144] <= lbus_di;
        //  16'h0134 : Key2[175:160] <= lbus_di;
        //  16'h0136 : Key2[191:176] <= lbus_di;
        //  16'h0138 : Key2[207:192] <= lbus_di;
        //  16'h013A : Key2[223:208] <= lbus_di;
        //  16'h013C : Key2[239:224] <= lbus_di;
        //  16'h013E : Key2[255:240] <= lbus_di;
         

        //  //Wots_Tree_index;  Wots_Tree_height;  Wots_KeyPair;  Sel_Tree;  Wots_Layer_Addr;
        //  16'h0140 : Wots_Tree_index     <= lbus_di;
        //  16'h0142 : Wots_Tree_height    <= lbus_di;
        //  16'h0144 : Wots_KeyPair        <= lbus_di;
        //  16'h0146 : Wots_Layer_Addr     <= lbus_di;
        //  16'h0148 : Sel_Tree[ 15:  0]   <= lbus_di;
        //  16'h014A : Sel_Tree[ 31: 16]   <= lbus_di;
        //  16'h014C : Sel_Tree[ 47: 32]   <= lbus_di;
        //  16'h014E : Sel_Tree[ 63: 48]   <= lbus_di;
        //  16'h014E : Sel_Tree[ 63: 48]   <= lbus_di;
         
        //  16'h0150 : Wots_Mode           <= lbus_di;
         
         

        endcase

      end
   end
                
   //------------------------------------------------
   always @(posedge clk or posedge rst)
     if (rst) 
       lbus_do <= 16'h0;
     else if (~lbus_rd)
       lbus_do <= mux_lbus_do(lbus_a, ctrl, Wots_Mode, blk_dout);
   
   function  [15:0] mux_lbus_do;
      input [15:0]   lbus_a;
      input [2:0]    ctrl;
      input [1:0]        Wots_Mode;
      input [127:0]  blk_dout;
      
      case(lbus_a)
        16'h0002: mux_lbus_do = ctrl;
        16'h000C: mux_lbus_do = Wots_Mode;
        16'h0180: mux_lbus_do = blk_dout_reg[127:112];
        16'h0182: mux_lbus_do = blk_dout_reg[111:96];
        16'h0184: mux_lbus_do = blk_dout_reg[95:80];
        16'h0186: mux_lbus_do = blk_dout_reg[79:64];
        16'h0188: mux_lbus_do = blk_dout_reg[63:48];
        16'h018A: mux_lbus_do = blk_dout_reg[47:32];
        16'h018C: mux_lbus_do = blk_dout_reg[31:16];
        16'h018E: mux_lbus_do = blk_dout_reg[15:0];
        16'hFFFC: mux_lbus_do = 16'h4702;
        default:  mux_lbus_do = 16'h0000;
      endcase
   endfunction
//    ila_1 the_ila(
//    .clk(clk),


//    .probe0(lbus_a),
//    .probe1(lbus_di),
//    .probe2(trig_wr),
//    .probe3(lbus_wr),
//    .probe4(Key1[63:0]),
//    .probe5(Wots_Mode)
//);
endmodule // LBUS_IF
