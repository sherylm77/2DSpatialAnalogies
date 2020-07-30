package main

import (
	"fmt"
	"image"
	"math"
	"math/rand"
	"testing"
)

func TestAngle(t *testing.T) {
	for i := 0; i < 50; i++ {
		Point := image.Point{}
		Point2 := image.Point{}
		Point.X = 4
		Point.Y = 4
		Size := 10
		for {
			Point2.X = rand.Intn(Size)
			Point2.Y = rand.Intn(Size)
			if Point2 != Point {
				break
			}
		}
		xDist := Point2.X - Point.X
		yDist := Point2.Y - Point.Y
		ang0 := 0.0
		ang360 := 0.0
		if xDist >= 0 && yDist >= 0 {
			ang0 = math.Atan2(float64(yDist), float64(xDist)) * 180 / math.Pi
		} else if xDist < 0 && yDist >= 0 {
			ang0 = math.Atan2(float64(yDist), float64(xDist)) * 180 / math.Pi
		} else if xDist >= 0 && yDist < 0 {
			ang360 = 360 - (math.Abs(math.Atan2(float64(yDist), float64(xDist))) * 180 / math.Pi)
		} else { //xDist < 0 and yDist < 0
			ang360 = 360 + (math.Atan2(float64(yDist), float64(xDist)) * 180 / math.Pi)
		}
		ang := ang0 + ang360
		fmt.Printf("%v %v %v %v %v %v \n", Point2, xDist, yDist, ang, ang0, ang360)
	}

}
