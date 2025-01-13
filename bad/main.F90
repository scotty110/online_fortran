program test_generate_points
    use generate_points
    use online_nn
    implicit none

    real(8), allocatable :: points(:,:)  ! 2D array to store x and y values
    integer :: n_points                  ! Number of points to generate
    integer :: i, j, epoch               ! Loop variables

    real(8), allocatable :: x(:), y(:), shuffled_x(:), shuffled_y(:)
    real(8), allocatable :: xp(:), yp(:), xi(:), yi(:), y_pred(:)
    real(8) :: loss
    integer, parameter :: max_epochs = 200

    n_points = 1000                      ! Number of points

    ! Call the subroutine to generate data
    call generate_data(points, n_points)

    ! Separate x and y values into individual arrays for convenience
    allocate(x(n_points), y(n_points))
    x = points(1, :)
    y = points(2, :)
    deallocate(points)

    ! Allocate arrays for xp, yp, xi, yi, and y_pred
    allocate(xp(1), yp(1), xi(1), yi(1), y_pred(1))

    ! Initialize xp and yp (previous x and y) to zero
    xp(1) = 0.0_8
    yp(1) = 0.0_8

    ! Training loop
    do epoch = 1, max_epochs
        ! Forward pass for each (xi, yi) and update xp, yp
        do i = 1, n_points
            xi(1) = x(i)
            yi(1) = y(i)

            ! Perform inference using the torch model
            call torch_inference(xi, xp, yp, y_pred)

            ! Compute loss (for demonstration purposes, using mean squared error)
            loss = (y_pred(1) - yi(1))**2

            ! Update xp and yp for the next step
            xp(1) = xi(1)
            yp(1) = yi(1)
        end do

        ! Evaluation phase: reset xp and yp to zero
        xp(1) = 0.0_8
        yp(1) = 0.0_8

        ! Evaluate model on the entire dataset
        loss = 0.0_8
        do i = 1, n_points
            xi(1) = x(i)
            yi(1) = y(i)

            ! Perform inference using the torch model
            call torch_inference(xi, xp, yp, y_pred)

            ! Accumulate the loss
            loss = loss + (y_pred(1) - yi(1))**2

            ! Update xp and yp for evaluation
            xp(1) = xi(1)
            yp(1) = yi(1)
        end do

        ! Calculate the average loss for evaluation
        loss = loss / real(n_points, 8)
        print *, "Epoch:", epoch, "Loss:", loss

        ! Shuffle x and y
        allocate(shuffled_x(n_points), shuffled_y(n_points))
        call shuffle_data(x, y, shuffled_x, shuffled_y)
        x = shuffled_x
        y = shuffled_y
        deallocate(shuffled_x, shuffled_y)
    end do

    ! Deallocate arrays
    deallocate(x, y, xp, yp, xi, yi, y_pred)

contains

    subroutine shuffle_data(x, y, shuffled_x, shuffled_y)
        implicit none
        real(8), intent(in) :: x(:), y(:)
        real(8), intent(out) :: shuffled_x(:), shuffled_y(:)
        integer :: n, i, idx
        integer, allocatable :: indices(:)

        n = size(x)
        allocate(indices(n))

        ! Generate random indices
        call random_seed()  ! Seed random number generator
        do i = 1, n
            indices(i) = i
        end do
        call shuffle_array(indices)

        ! Shuffle x and y based on indices
        do i = 1, n
            idx = indices(i)
            shuffled_x(i) = x(idx)
            shuffled_y(i) = y(idx)
        end do

        deallocate(indices)
    end subroutine shuffle_data

    subroutine shuffle_array(array)
        implicit none
        integer, intent(inout) :: array(:)
        integer :: i, temp, n
        real(8) :: rand_val

        n = size(array)
        do i = 1, n
            call random_number(rand_val)
            temp = 1 + int(rand_val * real(n, kind=8))
            temp = array(i)
            array(i) = array(temp)
            array(temp) = temp
        end do
    end subroutine shuffle_array

end program test_generate_points
