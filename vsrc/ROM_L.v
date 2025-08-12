

module ROM_L (
input [3:0]	a,
output reg [11:0] spo
);

always@(*)
begin
	case(a)
	4'd0: spo = 12'd0;
	4'd1: spo = 12'd767;
	4'd2: spo = 12'd1534;
	4'd3: spo = 12'd2301;
	4'd4: spo = 12'd3068;
	4'd5: spo = 12'd506;
	4'd6: spo = 12'd1273;
	4'd7: spo = 12'd2040;
	4'd8: spo = 12'd2807;
	4'd9: spo = 12'd245;
	4'd10: spo = 12'd1012;
	4'd11: spo = 12'd1779;
	4'd12: spo = 12'd2546;
	4'd13: spo = 12'd3313;	
	4'd14: spo = 12'd751;
	4'd15: spo = 12'd1518;		
	endcase
end

// ROM_H rom0 (
// .a (mul_h),
// .spo (lut_h)
// );

endmodule