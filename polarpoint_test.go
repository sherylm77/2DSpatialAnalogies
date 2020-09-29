package main

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/goki/mat32"
)

func TestPoints(t *testing.T) {
	Size := 9
	currentTime := time.Now()
	rand.Seed(int64(currentTime.Unix()))
	PointX := rand.Intn(Size)
	PointY := rand.Intn(Size)
	maxDist1 := math.Hypot(float64(9-PointX), float64(9-PointY)) // point 9, 9
	maxDist2 := math.Hypot(float64(PointX), float64(PointY))     // point 0, 0
	MaxDist := int(math.Min(maxDist1, maxDist2))
	MinDist := float32(4)
	dist := MinDist + rand.Float32()*(float32(MaxDist)-MinDist+1)
	ang := rand.Float32() * 360
	Point2X := int(math.Abs(float64(dist*mat32.Cos(ang*math.Pi/180)))) + PointX
	Point2Y := int(math.Abs(float64(dist*mat32.Sin(ang*math.Pi/180)))) + PointY
	hypotDist := float32(math.Hypot(float64(Point2X-PointX), float64(Point2Y-PointY)))
	actualAng := 180 * math.Atan2(float64(Point2Y-PointY), float64(Point2X-PointX)) / math.Pi
	fmt.Println("Point1: ", PointX, PointY)
	fmt.Println("Point2: ", Point2X, Point2Y)
	fmt.Println("Dist, , HypotDist, Ang, AngActual: ", dist, hypotDist, ang, actualAng)
	fmt.Println("Min, max1, max2, Max: ", MinDist, maxDist1, maxDist2, MaxDist)
}
