

module ROM_H (
input [3:0]	a,
output reg [11:0] spo
);

always@(*)
begin
	case(a)
	4'd0: spo = 12'd0;
	4'd1: spo = 12'd3270;
	4'd2: spo = 12'd3211;
	4'd3: spo = 12'd3152;
	4'd4: spo = 12'd3093;
	4'd5: spo = 12'd3034;
	4'd6: spo = 12'd2975;
	4'd7: spo = 12'd2916;
	4'd8: spo = 12'd2857;
	4'd9: spo = 12'd2798;
	4'd10: spo = 12'd2739;
	4'd11: spo = 12'd2680;
	4'd12: spo = 12'd2621;
	4'd13: spo = 12'd2562;	
	4'd14: spo = 12'd2503;
	4'd15: spo = 12'd2444;		
	endcase
end

// ROM_H rom0 (
// .a (mul_h),
// .spo (lut_h)
// );

endmodule