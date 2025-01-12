module generate_points
    implicit none
    contains

    subroutine generate_data(points, n_points)
        ! Subroutine to generate points for the equation y = 3.1x^3 + 2.2x^2 - 3.8x + 1.77

        real(8), allocatable, intent(out) :: points(:,:)  ! 2D array to hold x and y values
        integer, intent(in) :: n_points                  ! Number of points to generate

        real(8) :: x_min, x_max, dx                      ! Range and step size for x
        integer :: i                                     ! Loop variable

        x_min = -3.0_8                                   ! Start of the range
        x_max = 3.0_8                                    ! End of the range
        dx = (x_max - x_min) / real(n_points - 1, 8)     ! Step size for x

        ! Allocate the points array (2 rows: one for x, one for y)
        allocate(points(2, n_points))

        ! Generate points
        do i = 1, n_points
            points(1, i) = x_min + (i - 1) * dx          ! Calculate x value
            points(2, i) = 3.1 * points(1, i)**3 + 2.2 * points(1, i)**2 - &
                           3.8 * points(1, i) + 1.77    ! Calculate corresponding y value
        end do

    end subroutine generate_data

end module generate_points