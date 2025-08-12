module Div2(
    input  [11:0] Din,
    output [11:0] Dout
    );

wire[11:0]   lsb_data = Din[0]? 12'd1665 : 12'd0;
assign Dout = Din[11:1] + lsb_data;

endmodule

