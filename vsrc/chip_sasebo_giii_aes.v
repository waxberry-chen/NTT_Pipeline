/*-------------------------------------------------------------------------
 AES cryptographic module for FPGA on SASEBO-GIII
 
 File name   : chip_sasebo_giii_aes.v
 Version     : 1.0
 Created     : APR/02/2012
 Last update : APR/25/2013
 Desgined by : Toshihiro Katashita
 
 
 Copyright (C) 2012,2013 AIST
 
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
 appreciate it if you can cite our paper.
 (http://www.risec.aist.go.jp/project/sasebo/)
 -------------------------------------------------------------------------*/


//================================================ CHIP_SASEBO_GIII_AES
module CHIP_SASEBO_GIII_AES
  (// Local bus for GII
   lbus_di_a, //地址和数�??????
   lbus_do,   //
   lbus_wrn, lbus_rdn,
   lbus_clkn, lbus_rstn,

   // GPIO and LED
   gpio_startn, 
   gpio_endn, gpio_exec, led,   //这三个不用管

   // Clock OSC
   osc_en_b //晶振
   );
   
   //------------------------------------------------
   // Local bus for GII
   (* keep = "TRUE" *)input [15:0]  lbus_di_a;
   (* keep = "TRUE" *)output [15:0] lbus_do;
   (* keep = "TRUE" *)input         lbus_wrn, lbus_rdn;
   (* keep = "TRUE" *)input         lbus_clkn, lbus_rstn;

   // GPIO and LED
   (* keep = "TRUE" *)output        gpio_startn, gpio_endn, gpio_exec;
   (* keep = "TRUE" *)output [9:0]  led;

   // Clock OSC
   (* keep = "TRUE" *)output        osc_en_b;

   //------------------------------------------------
   // Internal clock
   (* keep = "TRUE" *)wire         clk, rst;

   // Local bus
   (* keep = "TRUE" *)reg [15:0]   lbus_a, lbus_di;
   
   // Block cipher
   (* keep = "TRUE" *)wire [127:0] blk_kin, blk_din, blk_dout;
   (* keep = "TRUE" *)wire         blk_krdy, blk_kvld, blk_drdy, blk_dvld;
   (* keep = "TRUE" *)wire         blk_encdec, blk_en, blk_rstn, blk_busy;
   (* keep = "TRUE" *)reg          blk_drdy_delay;
  
   //------------------------------------------------
   assign led[0] = rst;
   assign led[1] = lbus_rstn;
   assign led[2] = 1'b0;
   assign led[3] = blk_rstn;
   assign led[4] = blk_encdec;
   assign led[5] = blk_krdy;
   assign led[6] = blk_kvld;
   assign led[7] = 1'b0;
   assign led[8] = blk_dvld;
   assign led[9] = blk_busy;

   assign osc_en_b = 1'b0;
   //------------------------------------------------
   always @(posedge clk) if (lbus_wrn)  lbus_a  <= lbus_di_a;  //1 写地�??????
   always @(posedge clk) if (~lbus_wrn) lbus_di <= lbus_di_a;  //0 写数�??????

   (* keep = "TRUE" *)wire[11:0] a,b;

   LBUS_IF lbus_if
     (.lbus_a(lbus_a), .lbus_di(lbus_di), .lbus_do(lbus_do),
      .lbus_wr(lbus_wrn), .lbus_rd(lbus_rdn),
      .a(a),.b(b),
      .blk_dout(blk_dout),
      .blk_krdy(blk_krdy), .blk_drdy(blk_drdy), 
      .blk_kvld(blk_kvld), .blk_dvld(blk_dvld),
      .blk_en(blk_en), .blk_rstn(blk_rstn),
      .clk(clk), .rst(rst));

   //------------------------------------------------
   assign gpio_startn = ~blk_drdy;
   assign gpio_endn   = 1'b0; //~blk_dvld;
   assign gpio_exec   = 1'b0; //blk_busy;

   always @(posedge clk) blk_drdy_delay <= blk_drdy;

   // AES_Composite_enc AES_Composite_enc
   //   (.Kin(blk_kin), .Din(blk_din), .Dout(blk_dout),
   //    .Krdy(blk_krdy), .Drdy(blk_drdy_delay), .Kvld(blk_kvld), .Dvld(blk_dvld),
   //    /*.EncDec(blk_encdec),*/ .EN(blk_en), .BSY(blk_busy),
   //    .CLK(clk), .RSTn(blk_rstn));
   wire rst_n = ~rst;
//   (*KEEP_HIERARCHY = "{TRUE}" *) Haraka_Core u_Haraka_Core(
//       .clk              ( clk              ),
//       .rst_n            ( rst_n            ),
//       .Wots_Tree_index  ( Wots_Tree_index  ),
//       .Wots_Tree_height ( Wots_Tree_height ),
//       .Wots_KeyPair     ( Wots_KeyPair     ),
//       .Sel_Tree         ( Sel_Tree         ),
//       .Wots_Layer_Addr  ( Wots_Layer_Addr  ),
//       .Wots_Mode        ( Wots_Mode        ),
//       .Wots_Start       ( ~gpio_startn     )
//    );


//(* dont_touch = "true" *)(* keep = "TRUE" *) wire [11:0] product_reg;


    genvar i;
   generate
      for(i= 0; i<100; i=i+1) begin
      (* dont_touch = "true" *)(* keep = "TRUE" *) a_b_mul a_b_mul_inst(
         .clk   ( clk   ),
         .rst_n ( rst_n ),
         .a(a),
         .b(b),
         .start  ( ~gpio_startn )
      );
      end
   endgenerate
 


   //------------------------------------------------   
   MK_CLKRST mk_clkrst (.clkin(lbus_clkn), .rstnin(lbus_rstn),
                        .clk(clk), .rst(rst));
endmodule // CHIP_SASEBO_GIII_AES


   
//================================================ MK_CLKRST
module MK_CLKRST (clkin, rstnin, clk, rst);
   //synthesis attribute keep_hierarchy of MK_CLKRST is no;
   
   //------------------------------------------------
   input  clkin, rstnin;
   output clk, rst;
   
   //------------------------------------------------
   wire   refclk;
//   wire   clk_dcm, locked;

   //------------------------------------------------ clock
   IBUFG u10 (.I(clkin), .O(refclk)); 

/*
   DCM_BASE u11 (.CLKIN(refclk), .CLKFB(clk), .RST(~rstnin),
                 .CLK0(clk_dcm),     .CLKDV(),
                 .CLK90(), .CLK180(), .CLK270(),
                 .CLK2X(), .CLK2X180(), .CLKFX(), .CLKFX180(),
                 .LOCKED(locked));
   BUFG  u12 (.I(clk_dcm),   .O(clk));
*/

   BUFG  u12 (.I(refclk),   .O(clk));

   //------------------------------------------------ reset
   MK_RST u20 (.locked(rstnin), .clk(clk), .rst(rst));
endmodule // MK_CLKRST



//================================================ MK_RST
module MK_RST (locked, clk, rst);
   //synthesis attribute keep_hierarchy of MK_RST is no;
   
   //------------------------------------------------
   input  locked, clk;
   output rst;

   //------------------------------------------------
   reg [15:0] cnt;
   
   //------------------------------------------------
   always @(posedge clk or negedge locked) 
     if (~locked)    cnt <= 16'h0;
     else if (~&cnt) cnt <= cnt + 16'h1; // if cnt not full, ~&cnt=1, count up
   // when cnt full rst = 0
   assign rst = ~&cnt;
endmodule // MK_RST
